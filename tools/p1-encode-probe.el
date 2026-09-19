;;; p1-encode-probe.el --- P1: encode the Noul set, then probe it -*- lexical-binding: t; -*-

;; Three stages, selected by NSO_P1_STAGE (default "all"):
;;
;;   tokenize -- tokenise the prompts and report the length distribution.
;;               Costs nothing and decides the budget, since encoder time
;;               scales with sequence length.
;;   encode   -- run the frozen donor over every prompt with all 28 layers
;;               resident on the GPU, and write the per-position hidden states
;;               to build/p1-states.eld.  This is the expensive stage.
;;   probe    -- read those states, fit the three poolings at two depths, and
;;               write build/p1-results.org.  Cheap, and re-runnable without
;;               touching the GPU, which is the reason the states are written
;;               out at all.
;;
;; Progress goes to build/p1-progress.log as well as stdout, because batch
;; Emacs buffers stdout and a run this long needs to be watchable.

(defvar nso-p1--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(defun nso-p1--sib (name) (expand-file-name (concat "../../" name) nso-p1--here))
(defun nso-p1--own (name) (expand-file-name (concat "../" name) nso-p1--here))

(add-to-list 'load-path (nso-p1--own "lisp"))
(dolist (d '("nelisp-llm/lisp" "nelisp-photon/lisp" "nelisp-gpu/lisp"))
  (add-to-list 'load-path (nso-p1--sib d)))

(require 'nso-probe)

(defvar nso-p1--stage (or (getenv "NSO_P1_STAGE") "all"))
(defvar nso-p1--donor (nso-p1--sib "nelisp-llm/build/donor/qwen3-0.6b"))
(defvar nso-p1--build (nso-p1--own "build"))
(defvar nso-p1--states
  (or (getenv "NSO_P1_STATES")
      (expand-file-name "p1-states.eld" nso-p1--build))
  "Where encoded states are written and read.
Overridable so a control run on synthetic states can sit beside a real one
instead of overwriting the artefact that cost an hour of GPU time.")
(defvar nso-p1--results
  (or (getenv "NSO_P1_RESULTS")
      (expand-file-name "p1-results.org" nso-p1--build)))
(defvar nso-p1--log (expand-file-name "p1-progress.log" nso-p1--build))
(defvar nso-p1--mid-layer 13 "Zero-based index of the mid-depth capture point.")
(defvar nso-p1--progress-every
  (string-to-number (or (getenv "NSO_P1_PROGRESS") "10"))
  "Log progress every this many examples.
Set to 1 on a smoke run: batch Emacs buffers stdout, so if the process dies by
signal its stdout is lost entirely and the append-only progress file is the
only record of how far it got.")

(unless (file-directory-p nso-p1--build) (make-directory nso-p1--build t))

(defun nso-p1--say (fmt &rest args)
  (let ((line (concat (format-time-string "%H:%M:%S  ")
                      (apply #'format fmt args) "\n")))
    (princ line)
    (write-region line nil nso-p1--log t 'quiet)))

(defun nso-p1--rows (flat seq dim)
  "Split the flat SEQ x DIM vector FLAT into a list of per-position vectors."
  (let ((out nil) (i 0))
    (while (< i seq)
      (let ((v (make-vector dim 0.0)))
        (dotimes (j dim) (aset v j (aref flat (+ (* i dim) j))))
        (push v out))
      (setq i (1+ i)))
    (nreverse out)))

;;; --- stage: tokenize -----------------------------------------------------

(defvar nso-p1--data (nso-probe-load (nso-p1--own "data/noul-outcome.eld")))

(defun nso-p1-tokenize ()
  "Tokenise every prompt.  Returns a list of (EXAMPLE . IDS)."
  (require 'nl-llm-qwen-tokenizer)
  (let* ((tok (nl-llm-qwen-tok-load (expand-file-name "tokenizer.bin" nso-p1--donor)))
         (tmpl (plist-get nso-p1--data :template))
         (out nil) (total 0) (mx 0) (mn 9999))
    (dolist (e (plist-get nso-p1--data :examples))
      (let ((ids (nl-llm-qwen-tok-encode tok (nso-probe-prompt tmpl e))))
        (setq total (+ total (length ids))
              mx (max mx (length ids))
              mn (min mn (length ids)))
        (push (cons e ids) out)))
    (setq out (nreverse out))
    ;; NSO_P1_LIMIT truncates the set for a smoke run.  The data file lists
    ;; the two members of a pair adjacently, so an even limit keeps pairs
    ;; whole and the splitter's invariant holds.
    (let ((lim (getenv "NSO_P1_LIMIT")))
      (when lim
        (setq out (butlast out (max 0 (- (length out) (string-to-number lim)))))
        (setq total 0)
        (dolist (p out) (setq total (+ total (length (cdr p)))))
        (nso-p1--say "NSO_P1_LIMIT: truncated to %d prompts" (length out))))
    (nso-p1--say "tokenised %d prompts: min %d, mean %.1f, max %d tokens"
                 (length out) mn (/ (float total) (length out)) mx)
    (nso-p1--say "estimated encode time at 19.1s per 5 tokens: %.0f min"
                 (/ (* (/ 19.13 5.0) total) 60.0))
    out))

;;; --- stage: encode -------------------------------------------------------

(defun nso-p1-encode (tokenised)
  "Encode TOKENISED with all layers resident; write states to disk."
  (require 'nl-llm-weights)
  (require 'nl-llm-weights-forward)
  (unless (require 'nl-llm-weights-gpu nil t)
    (error "p1: nelisp-gpu is not loadable"))
  (nelisp-gpu-server-start)
  (unless (nelisp-gpu-server-up-p) (error "p1: the GPU server would not start"))
  (unwind-protect
      (let* ((wts (nl-llm-weights-open
                   (expand-file-name "weights.bin" nso-p1--donor)))
             (cfg (nl-llm-weights-config wts))
             (dim (plist-get cfg :dim))
             (nlayers (plist-get cfg :layers))
             (t0 (float-time))
             (layers nil))
        (dotimes (ly nlayers)
          (push (nl-llm-wgpu-load-layer wts ly) layers)
          ;; Logged in sevenths rather than only at the end: the upload is two
          ;; minutes of the run and a death inside it would otherwise leave no
          ;; trace of how far it got.
          (when (= 0 (mod (1+ ly) 7))
            (nso-p1--say "  uploaded %d/%d layers, %.0fs elapsed"
                         (1+ ly) nlayers (- (float-time) t0))))
        (setq layers (nreverse layers))
        (nso-p1--say "resident load: %.0fs for %d layers" (- (float-time) t0) nlayers)
        (unwind-protect
            (let ((rows nil) (i 0) (n (length tokenised)) (t1 (float-time)))
              (dolist (pair tokenised)
                (let* ((e (car pair))
                       (ids (cdr pair))
                       (seq (length ids))
                       (x (make-vector (* seq dim) 0.0))
                       (mid nil)
                       (p 0))
                  (dolist (tk ids)
                    (let ((row (nl-llm-weights-embed wts tk)))
                      (dotimes (j dim) (aset x (+ (* p dim) j) (aref row j))))
                    (setq p (1+ p)))
                  (let ((ly 0))
                    (dolist (lay layers)
                      (setq x (nl-llm-wgpu-block lay x seq cfg))
                      (when (= ly nso-p1--mid-layer)
                        (setq mid (copy-sequence x)))
                      (setq ly (1+ ly))))
                  ;; The same final RMSNorm at both depths, so a depth
                  ;; comparison is not confounded by the per-position scale
                  ;; the norm removes.
                  (push (list :pair (plist-get e :pair)
                              :label (plist-get e :label)
                              :hard (plist-get e :hard)
                              :text (plist-get e :text)
                              :seq seq
                              :mid (nso-p1--rows (nl-llm-wf-final-norm wts mid seq)
                                                 seq dim)
                              :final (nso-p1--rows (nl-llm-wf-final-norm wts x seq)
                                                   seq dim))
                        rows)
                  (setq i (1+ i))
                  (when (= 0 (mod i nso-p1--progress-every))
                    (let ((el (- (float-time) t1)))
                      (nso-p1--say "encoded %d/%d, %.0fs elapsed, %.0fs remaining"
                                   i n el (* (/ el i) (- n i)))))))
              (setq rows (nreverse rows))
              (nso-p1--say "writing states to %s" nso-p1--states)
              (with-temp-file nso-p1--states
                (let ((print-level nil) (print-length nil))
                  (prin1 (list :dim dim :mid-layer nso-p1--mid-layer :rows rows)
                         (current-buffer))))
              (nso-p1--say "states written (%.0f MB)"
                           (/ (float (nth 7 (file-attributes nso-p1--states)))
                              1048576.0))
              rows)
          (dolist (lay layers) (nl-llm-wgpu-free-layer lay))))
    (nelisp-gpu-server-stop)))

;;; --- stage: probe --------------------------------------------------------

(defun nso-p1--split-rows (rows)
  (let ((train nil) (test nil))
    (dolist (r rows)
      (if (= 0 (mod (plist-get r :pair) 3)) (push r test) (push r train)))
    (cons (nreverse train) (nreverse test))))

(defun nso-p1-probe (rows)
  "Fit every pooling at every depth over ROWS and write the report."
  (let* ((sp (nso-p1--split-rows rows))
         (train (car sp)) (test (cdr sp))
         (try (mapcar (lambda (r) (float (plist-get r :label))) train))
         (tey (mapcar (lambda (r) (float (plist-get r :label))) test))
         (report nil))
    (nso-p1--say "probe: %d train, %d held-out" (length train) (length test))
    (when (or (null test) (null train))
      (error (concat "p1: the split left one side empty (%d train, %d held-out). "
                     "Every third pair is held out, so a smoke run needs a limit "
                     "that reaches at least pair 3 on both sides")
             (length train) (length test)))
    (dolist (depth '(:final :mid))
      (dolist (kind '(last mean attn))
        ;; FEATURIZER turns a row into a feature vector and is refit on
        ;; whatever subset it is handed.  For `last' and `mean' it ignores
        ;; that subset; for `attn' it retrains the pool direction, so the
        ;; out-of-fold logits below come from a pooling that never saw the
        ;; fold either.  Fitting the pool once on all of train and only
        ;; refitting the head moved the leak instead of removing it: on noise
        ;; features that combination still returned T=0.21 and an ECE of
        ;; 0.346, while the two fixed poolings reached 0.011 and 0.073.
        (let* ((featurizer
                (lambda (fit-rows fit-ys)
                  (let ((u (when (eq kind 'attn)
                             (plist-get (nso-attn-train
                                         (mapcar (lambda (r) (plist-get r depth))
                                                 fit-rows)
                                         fit-ys 400 0.5 0.05)
                                        :u))))
                    (lambda (r)
                      (let ((states (plist-get r depth)))
                        (cond ((eq kind 'last) (nso-pool-last states))
                              ((eq kind 'mean) (nso-pool-mean states))
                              ((eq kind 'attn) (car (nso-pool-attn u states)))
                              (t (error "p1: unknown pooling %S" kind))))))))
               ;; The reported model is fitted on the whole training split.
               (feat (funcall featurizer train try))
               (trx (mapcar feat train))
               (tex (mapcar feat test))
               (fit (nso-probe-fit-and-score trx try tex tey 600 0.5 0.05 5))
               (s (plist-get fit :test))
               (te-logits (plist-get fit :logits))
               ;; Temperature from out-of-fold logits: each training example
               ;; scored by a head AND a pooling that never saw its pair.  Not
               ;; held-out, which would calibrate against the reported set, and
               ;; not the training logits, which the head has already
               ;; separated -- that was this file's first version and on noise
               ;; it produced T=0.13 with held-out ECE going 0.169 -> 0.451.
               (tfit (nso-temperature-fit
                      (nso-probe-oof-logits
                       train try
                       (mapcar (lambda (r) (plist-get r :pair)) train)
                       featurizer 3 600 0.5 0.05)
                      try))
               (temp (plist-get tfit :temperature))
               (scaled (mapcar (lambda (z) (nso-sigmoid (/ z temp))) te-logits))
               (s-after (nso-probe-score scaled tey 5))
               (easy (nso-probe-subset (mapcar #'nso-sigmoid te-logits) tey test
                                       (lambda (r) (not (plist-get r :hard)))))
               (hard (nso-probe-subset (mapcar #'nso-sigmoid te-logits) tey test
                                       (lambda (r) (plist-get r :hard)))))
          (push (list :depth depth :kind kind :temp temp
                      :saturated (plist-get tfit :saturated)
                      :train (plist-get fit :train) :test s :after s-after
                      :easy (nso-probe-score (nth 0 easy) (nth 1 easy) 5)
                      :hard (nso-probe-score (nth 0 hard) (nth 1 hard) 5))
                report)
          (nso-p1--say "%s/%s: held-out acc %.3f, ECE %.3f -> %.3f (T=%.2f%s)"
                       depth kind (plist-get s :accuracy)
                       (plist-get s :ece) (plist-get s-after :ece) temp
                       (if (plist-get tfit :saturated) ", AT BOUND" "")))))
    (setq report (nreverse report))
    (nso-p1--write-report report (length train) (length test))
    report))

(defun nso-p1--cell (s)
  "One table cell for score plist S, or a dash when the subset is empty."
  (if (plist-get s :empty) "-- (n=0)"
    (format "%.3f (n=%d)" (plist-get s :accuracy) (plist-get s :n))))

(defun nso-p1--write-report (report ntrain ntest)
  (with-temp-file nso-p1--results
    (insert "#+TITLE: P1 results -- frozen Qwen3-0.6B as a Noul encoder\n")
    (insert (format "#+DATE: %s\n\n" (format-time-string "%Y-%m-%d %H:%M")))
    (insert (format "Dataset: 140 minimal-pair examples, %d train / %d held-out,\n"
                    ntrain ntest))
    (insert "split by pair.  Majority-class baseline 0.500.  Unigram baseline at\n")
    (insert "chance out of fold (see test/probe-test.el).\n")
    (insert "A temperature marked * rested on the edge of the search range: the\n")
    (insert "optimum lies past the bound, so the figure is the bound, not a fit.\n\n")
    (insert "| depth | pooling | train acc | held-out acc | 95% CI | ECE | ECE after T | T |\n")
    (insert "|-------+---------+-----------+--------------+--------+-----+-------------+---|\n")
    (dolist (r report)
      (let ((s (plist-get r :test)) (a (plist-get r :after)))
        (insert (format "| %s | %s | %.3f | %.3f | [%.3f,%.3f] | %.3f | %.3f | %s |\n"
                        (substring (symbol-name (plist-get r :depth)) 1)
                        (plist-get r :kind)
                        (plist-get (plist-get r :train) :accuracy)
                        (plist-get s :accuracy)
                        (plist-get s :ci-lo) (plist-get s :ci-hi)
                        (plist-get s :ece) (plist-get a :ece)
                        (if (plist-get r :saturated)
                            (format "%.2f*" (plist-get r :temp))
                          (format "%.2f" (plist-get r :temp)))))))
    (insert "\n| depth | pooling | antonym acc | compositional acc |\n")
    (insert "|-------+---------+-------------+-------------------|\n")
    (dolist (r report)
      (insert (format "| %s | %s | %s | %s |\n"
                      (substring (symbol-name (plist-get r :depth)) 1)
                      (plist-get r :kind)
                      (nso-p1--cell (plist-get r :easy))
                      (nso-p1--cell (plist-get r :hard))))))
  (nso-p1--say "report written to %s" nso-p1--results))

;;; --- driver --------------------------------------------------------------

;; The error is written to the progress file before it is re-signalled.
;; Batch Emacs buffers stdout, so a run redirected to a file and then killed --
;; or one that dies after the buffer was last flushed -- loses its message
;; entirely, and two failures in this file were diagnosed only by where the
;; progress log stopped rather than by what it said.  `write-region' appends
;; immediately, so this survives what stdout does not.

(nso-p1--say "stage: %s" nso-p1--stage)
(condition-case err
    (cond
     ((equal nso-p1--stage "tokenize")
      (nso-p1-tokenize))
     ((equal nso-p1--stage "probe")
      (let ((saved (with-temp-buffer
                     (insert-file-contents nso-p1--states)
                     (read (buffer-string)))))
        (nso-p1-probe (plist-get saved :rows))))
     (t
      (let ((tk (nso-p1-tokenize)))
        (nso-p1-probe (nso-p1-encode tk)))))
  (error
   (nso-p1--say "FAILED: %s" (error-message-string err))
   (signal (car err) (cdr err))))

(nso-p1--say "done")

;;; p1-encode-probe.el ends here
