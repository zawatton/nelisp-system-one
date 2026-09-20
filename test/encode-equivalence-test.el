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
             ;; A sweep, not one length.  The single seq-5 case passed while
             ;; the P2 encode died at seq 14 with an index of exactly
             ;; seq*ff -- one past the end of the FFN buffer.  A batched path
             ;; is all strides and offsets, and those are what a single
             ;; length cannot exercise.
             (seqs (list 1 2 5 13 14 16))
             (lay (nl-llm-wgpu-load-layer wts 0))
)
        (unwind-protect
            (let ((total-slow 0.0) (total-fast 0.0))
              (dolist (seq seqs)
                (let* ((x (let ((v (make-vector (* seq dim) 0.0)))
                            (dotimes (n (* seq dim))
                              (aset v n (* 0.37 (- (mod (* (1+ n) 7919) 211) 105))))
                            v))
                       (t0 (float-time))
                       (slow (nl-llm-wgpu-block lay (copy-sequence x) seq cfg))
                       (t1 (float-time))
                       (fast (condition-case err
                                 (nso-encode-block lay (copy-sequence x) seq cfg)
                               (error (list :error (error-message-string err)))))
                       (t2 (float-time))
                       (diffs 0) (worst 0.0) (nan 0))
                  (setq total-slow (+ total-slow (- t1 t0))
                        total-fast (+ total-fast (- t2 t1)))
                  (if (and (consp fast) (eq (car fast) :error))
                      (nso-t (format "seq %d: the batched path runs at all" seq)
                             nil (nth 1 fast))
                    (nso-t (format "seq %d: same shape" seq)
                           (= (length slow) (length fast)))
                    (dotimes (n (min (length slow) (length fast)))
                      (let ((a (aref slow n)) (b (aref fast n)))
                        (when (or (/= a a) (/= b b)) (setq nan (1+ nan)))
                        (unless (= a b)
                          (setq diffs (1+ diffs))
                          (setq worst (max worst (abs (- a b)))))))
                    (nso-t (format "seq %d: no NaN" seq) (= 0 nan))
                    (nso-t (format "seq %d: bit-identical (%d/%d differ, worst %g)"
                                   seq diffs (length slow) worst)
                           (= 0 diffs)))))
              (message "  slow %.2fs, fast %.2fs over %d lengths"
                       total-slow total-fast (length seqs))
              (nso-t-lt "and the batched path is faster overall"
                        total-fast (* 0.8 total-slow)))
          (nl-llm-wgpu-free-layer lay)))
    (nelisp-gpu-server-stop))
  (nso-t-done "encode equivalence")))

;;; encode-equivalence-test.el ends here
