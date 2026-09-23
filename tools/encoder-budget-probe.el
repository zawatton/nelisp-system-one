;;; encoder-budget-probe.el --- what one encoder pass costs -*- lexical-binding: t; -*-

;; P1 needs a frozen Qwen3-0.6B forward per example.  Before choosing a dataset
;; size, measure what one pass costs, because the sibling repository's own
;; `nl-llm-wgpu-next-token' loads and frees each layer per call -- it is written
;; to keep peak VRAM at one layer -- and therefore pays the upload 28 times per
;; example.  Its docstring puts a per-block upload at about five seconds, which
;; would be over two minutes of transfer before any arithmetic.
;;
;; The question this probe answers: if the 28 layers are uploaded ONCE and kept
;; resident, what is the marginal cost of an example?  That number decides
;; whether P1 gets hundreds of examples or dozens.
;;
;; Run:  emacs -Q --batch -l tools/encoder-budget-probe.el
;; NSO_PROBE_EXAMPLES and NSO_PROBE_TOKENS override the defaults.

(defvar nso-probe--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(defun nso-probe--sib (name)
  (expand-file-name (concat "../../" name) nso-probe--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)

(require 'photon-tensor)
(require 'nl-llm-weights)
(require 'nl-llm-weights-forward)

(defvar nso-probe--table
  (or (getenv "NSO_DONOR")
      (nso-probe--sib "nelisp-llm/build/donor/qwen3-0.6b/weights.bin")))

(defvar nso-probe--examples
  (string-to-number (or (getenv "NSO_PROBE_EXAMPLES") "3")))
(defvar nso-probe--tokens
  (string-to-number (or (getenv "NSO_PROBE_TOKENS") "5")))

(defun nso-probe--say (fmt &rest args)
  (princ (apply #'format fmt args))
  (princ "\n"))

(defun nso-probe--embed (wts tokens dim)
  (let ((x (make-vector (* (length tokens) dim) 0.0)) (i 0))
    (dolist (tk tokens)
      (let ((row (nl-llm-weights-embed wts tk)))
        (dotimes (j dim) (aset x (+ (* i dim) j) (aref row j))))
      (setq i (1+ i)))
    x))

(if (not (file-readable-p nso-probe--table))
    (nso-probe--say "SKIP: donor table missing at %s" nso-probe--table)
  (if (not (require 'nl-llm-weights-gpu nil t))
      (nso-probe--say "SKIP: nelisp-gpu is not loadable")
    (let ((up (ignore-errors (nelisp-gpu-server-start)
                             (nelisp-gpu-server-up-p))))
      (if (not up)
          (nso-probe--say "SKIP: the GPU server would not start")
        (unwind-protect
            (let* ((wts (nl-llm-weights-open nso-probe--table))
                   (cfg (nl-llm-weights-config wts))
                   (dim (plist-get cfg :dim))
                   (nlayers (plist-get cfg :layers))
                   ;; A real Qwen token id sequence, borrowed from the sibling
                   ;; suite so the arithmetic is on plausible activations.
                   (base '(785 6722 315 9625 374 264 3283 304 9625 323))
                   (tokens (let ((out nil) (i 0))
                             (while (< i nso-probe--tokens)
                               (push (nth (mod i (length base)) base) out)
                               (setq i (1+ i)))
                             (nreverse out)))
                   (layers nil)
                   (t0 (float-time)))
              (nso-probe--say "model: %d layers, dim %d, seq %d, %d examples"
                              nlayers dim (length tokens) nso-probe--examples)
              ;; --- upload every layer once ---------------------------------
              (dotimes (ly nlayers)
                (push (nl-llm-wgpu-load-layer wts ly) layers)
                (when (= 0 (mod (1+ ly) 7))
                  (nso-probe--say "  uploaded %d/%d layers, %.1fs elapsed"
                                  (1+ ly) nlayers (- (float-time) t0))))
              (setq layers (nreverse layers))
              (let ((load-secs (- (float-time) t0)))
                (nso-probe--say "RESIDENT LOAD: %.1fs for %d layers"
                                load-secs nlayers)
                (unwind-protect
                    (let ((times nil) (e 0))
                      (while (< e nso-probe--examples)
                        (let* ((t1 (float-time))
                               (x (nso-probe--embed wts tokens dim)))
                          (dolist (lay layers)
                            (setq x (nl-llm-wgpu-block lay x (length tokens) cfg)))
                          (let ((secs (- (float-time) t1)))
                            (push secs times)
                            (nso-probe--say "  example %d: %.2fs  (hidden[0]=%.6f)"
                                            e secs (aref x 0))))
                        (setq e (1+ e)))
                      (setq times (nreverse times))
                      (let ((sum 0.0) (best 1.0e30))
                        (dolist (s times)
                          (setq sum (+ sum s) best (min best s)))
                        (nso-probe--say
                         (concat "\nPER-EXAMPLE (resident): mean %.2fs  best %.2fs"
                                 "  -- %d tokens, %d layers")
                         (/ sum (length times)) best (length tokens) nlayers)
                        (nso-probe--say
                         "BUDGET: 100 examples = %.1f min, 1000 = %.1f h"
                         (/ (* 100 (/ sum (length times))) 60.0)
                         (/ (* 1000 (/ sum (length times))) 3600.0))))
                  (dolist (lay layers) (nl-llm-wgpu-free-layer lay)))))
          (nelisp-gpu-server-stop))))))

;;; encoder-budget-probe.el ends here
