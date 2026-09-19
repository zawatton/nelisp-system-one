;;; nso-encode.el --- one imported block per round trip, not one per position -*- lexical-binding: t; -*-

;;; Commentary:

;; A faster driver for the sibling repository's GPU block.  It computes exactly
;; the same thing; what changes is how many times Emacs talks to the vkserver.
;;
;; Measured on this hardware before the change: 39.7s to encode one 9-token
;; example through 28 layers.  The arithmetic is not the cost.  Each layer
;; applies seven matrices, `nl-llm-wgpu-block' loops over positions and calls
;; `nl-llm-wgpu-apply' once per (layer, role, position), and each of those
;; calls is three IPC round trips -- upload the activation, run the kernel,
;; free the buffer.  That is 28 x 7 x 9 x 3 = 5292 round trips per example, and
;; 39.7s / 5292 is 7.5ms each.  The GTX 1060 does a 1024x1024 int8 matmul in
;; microseconds; the pipe does not.
;;
;; The sibling repository already ships the fix and does not use it here:
;; `nl-llm-wgpu-apply-seq' carries every position in one round trip and its
;; docstring states it is bit-identical to calling `nl-llm-wgpu-apply' SEQ
;; times.  It is wired into the backward and the resident path, not into the
;; forward block.  So this file restructures the caller -- gather each stage's
;; activations into one contiguous buffer, issue one batched call per role --
;; and leaves every piece of arithmetic to the same functions as before:
;; `nl-llm-wf--rmsnorm', `nl-llm--rmsnorm-heads', `nl-llm--rope-heads',
;; `nl-llm-wf--attend', `nl-llm-wf--silu-mul'.  The parts that were hard to get
;; right -- the decoupled head width, the half-split rotation, QK-norm -- are
;; not touched, and not copied either.
;;
;; Round trips per layer fall from 7 x SEQ x 3 to 7 x 3.  Because the claim is
;; bit-identity rather than approximation, it is checkable exactly, and
;; `test/encode-equivalence-test.el' checks it against the original on a real
;; layer: any difference at all is a failure, not a tolerance.
;;
;; These are another package's private functions (`nl-llm-wf--rmsnorm' and
;; friends).  Reaching for them is deliberate -- reimplementing them here would
;; be the version of this change that silently diverges -- and the equivalence
;; test is what makes the reach safe.

;;; Code:

(require 'photon-tensor)
(require 'nl-llm-attn)
(require 'nl-llm-weights)
(require 'nl-llm-weights-forward)
(require 'nl-llm-weights-gpu)

;; Two sibling files locate binaries relative to their own source -- the
;; vkserver in `nl-llm-gpu.el', the vkrun path in `photon-tensor-gpu.el' --
;; by expanding "../../nelisp-gpu/..." against `load-file-name'.  That holds
;; while they load from their own tree and breaks the moment a byte-compiled
;; copy lives anywhere else, which is what `tools/compile-deps.el' arranges:
;; the path resolves outside the workspace and the server start fails with
;; "Doing vfork: No such file or directory".
;;
;; Pinning the binary here, from THIS file's location, fixes it for whichever
;; of them is compiled, instead of excluding files one at a time and
;; rediscovering the class the next time one is added.

(defvar nso-encode--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(let ((bin (expand-file-name "../../nelisp-gpu/host/vkserver" nso-encode--here)))
  (when (file-executable-p bin)
    (setq nelisp-gpu-server-bin bin)))

(defun nso-encode--gather (src seq width fn)
  "Build a SEQ x WIDTH buffer by calling FN with each position index.
FN returns that position's WIDTH-long vector."
  (let ((out (make-vector (* seq width) 0.0))
        (i 0))
    (while (< i seq)
      (let ((row (funcall fn i)))
        (dotimes (t0 width) (aset out (+ (* i width) t0) (aref row t0))))
      (setq i (1+ i)))
    (ignore src)
    out))

(defun nso-encode--lin-seq (lay role x seq stride)
  "Apply LAY's ROLE matrix to every position of X in one round trip."
  (nl-llm-wgpu-apply-seq (plist-get (nl-llm-wgpu-layer-lins lay) role)
                         (plist-get (nl-llm-wgpu-layer-handles lay) role)
                         x 0 seq stride))

(defun nso-encode-block (lay x seq cfg)
  "Run one imported block over X (flat SEQ x dim), batching each role's matmul.
Bit-identical to `nl-llm-wgpu-block'; see this file's commentary."
  (let* ((dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (ff (nl-llm-weights-lin-rows
              (plist-get (nl-llm-wgpu-layer-lins lay) :wg)))
         ;; --- attention input: normalise every position, then three calls ---
         (a (nso-encode--gather
             x seq dim
             (lambda (i) (nl-llm-wf--rmsnorm x (* i dim) dim
                                             (nl-llm-wgpu-layer-ln1g lay) eps))))
         (q (nso-encode--lin-seq lay :wq a seq dim))
         (k (nso-encode--lin-seq lay :wk a seq dim))
         (v (nso-encode--lin-seq lay :wv a seq dim)))
    ;; --- per-head norms and the rotation: unchanged, on the CPU ------------
    (dotimes (i seq)
      (nl-llm--rmsnorm-heads q (* i qdim) heads hd
                             (photon-tensor (list hd)
                                            (nl-llm-wgpu-layer-q-norm lay))
                             eps)
      (nl-llm--rmsnorm-heads k (* i kvdim) kv-heads hd
                             (photon-tensor (list hd)
                                            (nl-llm-wgpu-layer-k-norm lay))
                             eps)
      (nl-llm--rope-heads q (* i qdim) heads hd i rbase 'half)
      (nl-llm--rope-heads k (* i kvdim) kv-heads hd i rbase 'half))
    (let* ((ctx (nl-llm-wf--attend q k v seq heads kv-heads hd))
           ;; ctx is already SEQ x qdim and contiguous, so wo needs no gather.
           (o (nso-encode--lin-seq lay :wo ctx seq qdim))
           (x1 (make-vector (* seq dim) 0.0)))
      (dotimes (n (* seq dim))
        (aset x1 n (+ (aref x n) (aref o n))))
      (let* ((b (nso-encode--gather
                 x1 seq dim
                 (lambda (i) (nl-llm-wf--rmsnorm x1 (* i dim) dim
                                                 (nl-llm-wgpu-layer-ln2g lay) eps))))
             (g (nso-encode--lin-seq lay :wg b seq dim))
             (u (nso-encode--lin-seq lay :wu b seq dim))
             ;; SwiGLU per position, gathered into one SEQ x ff buffer so the
             ;; down projection is also a single call.
             (h (let ((buf (make-vector (* seq ff) 0.0)))
                  (dotimes (i seq)
                    (let ((gi (make-vector ff 0.0)) (ui (make-vector ff 0.0)))
                      (dotimes (t0 ff)
                        (aset gi t0 (aref g (+ (* i ff) t0)))
                        (aset ui t0 (aref u (+ (* i ff) t0))))
                      (let ((hi (nl-llm-wf--silu-mul gi ui ff)))
                        (dotimes (t0 ff)
                          (aset buf (+ (* i ff) t0) (aref hi t0))))))
                  buf))
             (d (nso-encode--lin-seq lay :wd h seq ff))
             (out (make-vector (* seq dim) 0.0)))
        (dotimes (n (* seq dim))
          (aset out n (+ (aref x1 n) (aref d n))))
        out))))

(provide 'nso-encode)
;;; nso-encode.el ends here
