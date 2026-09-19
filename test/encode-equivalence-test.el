;;; encode-equivalence-test.el --- the fast block must be the same block -*- lexical-binding: t; -*-

;;; Commentary:

;; `nso-encode-block' exists only to issue fewer IPC round trips than
;; `nl-llm-wgpu-block'.  It claims to compute the same thing, and the claim is
;; bit-identity rather than approximation, because `nl-llm-wgpu-apply-seq'
;; states bit-identity with SEQ calls of `nl-llm-wgpu-apply'.  So the check is
;; exact equality -- any difference at all is a failure, and a tolerance here
;; would be a way of not noticing a reordered index.
;;
;; The comparison runs on a real imported layer with real weights, not a
;; synthetic one: the restructuring is about strides and offsets, and a
;; synthetic layer with dim == qdim == ff would let a stride mix-up pass.
;;
;; Needs the donor table and a Vulkan device; skips cleanly without either,
;; and says which is missing rather than passing silently.

;;; Code:

(defvar nso-eq--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(dolist (d '("nelisp-llm/lisp" "nelisp-photon/lisp" "nelisp-gpu/lisp"))
  (add-to-list 'load-path (expand-file-name (concat "../../" d) nso-eq--here)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-eq--here))

(load (expand-file-name "nso-test-helper.el" nso-eq--here))

(defvar nso-eq--table
  (expand-file-name "../../nelisp-llm/build/donor/qwen3-0.6b/weights.bin"
                    nso-eq--here))

(message "== encode equivalence ==")

(cond
 ;; Gated so `make test' stays fast, deterministic and free of any dependency
 ;; on weights or a GPU.  Run it with NSO_GPU_TESTS=1 when the encoder changes.
 ((not (getenv "NSO_GPU_TESTS"))
  (message "  SKIP: set NSO_GPU_TESTS=1 to run (needs the donor table and a GPU)")
  (message "encode equivalence: skipped"))
 ((not (file-readable-p nso-eq--table))
  (message "  SKIP: donor table missing (%s)" nso-eq--table)
  (message "encode equivalence: skipped"))
 ((not (require 'nso-encode nil t))
  (message "  SKIP: nso-encode is not loadable (nelisp-gpu absent?)")
  (message "encode equivalence: skipped"))
 ((not (ignore-errors (nelisp-gpu-server-start) (nelisp-gpu-server-up-p)))
  (message "  SKIP: the GPU server would not start")
  (message "encode equivalence: skipped"))
 (t
  (unwind-protect
      (let* ((wts (nl-llm-weights-open nso-eq--table))
             (cfg (nl-llm-weights-config wts))
             (dim (plist-get cfg :dim))
             (seq 5)
             (lay (nl-llm-wgpu-load-layer wts 0))
             ;; Deterministic input, spread over a plausible activation range.
             (x (let ((v (make-vector (* seq dim) 0.0)))
                  (dotimes (n (* seq dim))
                    (aset v n (* 0.37 (- (mod (* (1+ n) 7919) 211) 105))))
                  v)))
        (unwind-protect
            (let* ((t0 (float-time))
                   (slow (nl-llm-wgpu-block lay (copy-sequence x) seq cfg))
                   (t1 (float-time))
                   (fast (nso-encode-block lay (copy-sequence x) seq cfg))
                   (t2 (float-time))
                   (diffs 0)
                   (worst 0.0)
                   (nan 0))
              (nso-t "both paths return the same shape"
                     (= (length slow) (length fast)))
              (dotimes (n (min (length slow) (length fast)))
                (let ((a (aref slow n)) (b (aref fast n)))
                  (when (or (/= a a) (/= b b)) (setq nan (1+ nan)))
                  (unless (= a b)
                    (setq diffs (1+ diffs))
                    (setq worst (max worst (abs (- a b)))))))
              (message "  slow %.2fs, fast %.2fs, speedup %.1fx"
                       (- t1 t0) (- t2 t1) (/ (- t1 t0) (max 1e-9 (- t2 t1))))
              (message "  elements differing: %d of %d (worst %g), NaN %d"
                       diffs (length slow) worst nan)
              (nso-t "no NaN on either side" (= 0 nan))
              (nso-t "every element is bit-identical" (= 0 diffs))
              ;; The point of the change.  A fast path that is not faster is a
              ;; complication, so this is a check and not a note.
              (nso-t-lt "and the batched path is actually faster"
                        (- t2 t1) (* 0.8 (- t1 t0))))
          (nl-llm-wgpu-free-layer lay)))
    (nelisp-gpu-server-stop))
  (nso-t-done "encode equivalence")))

;;; encode-equivalence-test.el ends here
