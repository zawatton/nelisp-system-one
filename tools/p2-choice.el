;;; p2-choice.el --- P2: encode the intent set, then choose among unseen options -*- lexical-binding: t; -*-

;; Stages, selected by NSO_P2_STAGE (default "all"): tokenize / encode / probe.
;; Same shape as tools/p1-encode-probe.el and for the same reasons -- states
;; are written to disk so the probe can be re-run without the GPU, progress is
;; appended to a file because batch Emacs buffers stdout, and a failure is
;; recorded before it is re-signalled.
;;
;; What it measures is the acceptance criterion for P2: the head is fitted on
;; eight topics and then asked to choose among the four it has never seen --
;; neither their sentences nor their option text were available while fitting.

(defvar nso-p2--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(defun nso-p2--sib (name) (expand-file-name (concat "../../" name) nso-p2--here))
(defun nso-p2--own (name) (expand-file-name (concat "../" name) nso-p2--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(add-to-list 'load-path (nso-p2--own "build/elc"))

(require 'nso-choice)
(require 'nso-probe)

(defvar nso-p2--stage (or (getenv "NSO_P2_STAGE") "all"))
(defvar nso-p2--donor (nso-p2--sib "nelisp-llm/build/donor/qwen3-0.6b"))
(defvar nso-p2--build (nso-p2--own "build"))
(defvar nso-p2--states (or (getenv "NSO_P2_STATES")
                           (expand-file-name "p2-states.eld" nso-p2--build)))
(defvar nso-p2--results (or (getenv "NSO_P2_RESULTS")
                            (expand-file-name "p2-results.org" nso-p2--build)))
(defvar nso-p2--log (expand-file-name "p2-progress.log" nso-p2--build))
(defvar nso-p2--mid-layer 13)
(defvar nso-p2--checkpoint 10)

(unless (file-directory-p nso-p2--build) (make-directory nso-p2--build t))

(defun nso-p2--say (fmt &rest args)
  (let ((line (concat (format-time-string "%H:%M:%S  ") (apply #'format fmt args) "\n")))
    (princ line)
    (write-region line nil nso-p2--log t 'quiet)))

(defvar nso-p2--data
  (with-temp-buffer
    (insert-file-contents (nso-p2--own "data/choice-intent.eld"))
    (read (buffer-string))))

(defun nso-p2--topics () (append (plist-get nso-p2--data :topics) nil))
(defun nso-p2--examples () (append (plist-get nso-p2--data :examples) nil))

(defun nso-p2--held-p (name)
  (let (res)
    (dolist (tp (nso-p2--topics))
      (when (equal name (plist-get tp :name)) (setq res (plist-get tp :held))))
    res))

;;; --- what has to be encoded ------------------------------------------------
;;
;; Sentences go through the template; topic names are encoded bare, because an
;; option is a label rather than a request.

(defvar nso-p2--option-form (or (getenv "NSO_P2_OPTION") "description")
  "Which text stands for an option: `label' or `description'.
Both are encoded, so the two can be compared from one states file without
paying for the encoder twice.")

(defun nso-p2--option-text (name)
  "The text whose embedding represents the option NAME."
  (if (equal nso-p2--option-form "label")
      name
    (let (res)
      (dolist (tp (nso-p2--topics))
        (when (equal name (plist-get tp :name))
          (setq res (or (plist-get tp :description) name))))
      res)))

(defun nso-p2--items ()
  "Every text to encode, as (KIND NAME TEXT)."
  (let ((tmpl (plist-get nso-p2--data :template)) (out nil))
    (dolist (tp (nso-p2--topics))
      ;; Both forms, so a run can switch between them without re-encoding.
      (push (list 'option (plist-get tp :name) (plist-get tp :name)) out)
      (when (plist-get tp :description)
        (push (list 'option (plist-get tp :name) (plist-get tp :description)) out)))
    (dolist (e (nso-p2--examples))
      (push (list 'state (plist-get e :topic) (format tmpl (plist-get e :text))) out))
    (nreverse out)))

(defun nso-p2--cached ()
  (when (and (file-readable-p nso-p2--states) (not (getenv "NSO_P2_FORCE")))
    (plist-get (with-temp-buffer
                 (insert-file-contents nso-p2--states)
                 (read (buffer-string)))
               :rows)))

(defun nso-p2--write (rows dim)
  (with-temp-file nso-p2--states
    (let ((print-level nil) (print-length nil))
      (prin1 (list :dim dim :mid-layer nso-p2--mid-layer :rows rows) (current-buffer)))))

(defun nso-p2-encode ()
  "Encode every item; reuse anything already on disk."
  (require 'nl-llm-qwen-tokenizer)
  (require 'nl-llm-weights)
  (require 'nl-llm-weights-forward)
  (unless (require 'nl-llm-weights-gpu nil t) (error "p2: nelisp-gpu is not loadable"))
  (require 'nso-encode)
  ;; Before the 130-second resident load, not after: the GPU is shared with
  ;; whatever else the machine is doing, and a previous run died two minutes
  ;; in with "Process vkserver not running: terminated" while Ollama held 3.1
  ;; GB of the 6 GB card.  The symptom said nothing about the cause.
  (nso-p2--say "free VRAM: %s MiB" (or (nso-encode-free-vram-mib) "unknown"))
  (nso-encode-check-vram)
  (let* ((tok (nl-llm-qwen-tok-load (expand-file-name "tokenizer.bin" nso-p2--donor)))
         (items (nso-p2--items))
         (cached (nso-p2--cached))
         (have (let ((h (make-hash-table :test 'equal)))
                 (dolist (r cached) (puthash (plist-get r :text) r h))
                 h))
         (todo (let (out)
                 (dolist (it items)
                   (unless (gethash (nth 2 it) have) (push it out)))
                 (nreverse out))))
    (nso-p2--say "cache: %d reused, %d to encode (%.0f min)"
                 (length cached) (length todo) (/ (* 10.0 (length todo)) 60.0))
    (if (null todo)
        cached
      (nelisp-gpu-server-start)
      (unless (nelisp-gpu-server-up-p) (error "p2: the GPU server would not start"))
      (unwind-protect
          (let* ((wts (nl-llm-weights-open (expand-file-name "weights.bin" nso-p2--donor)))
                 (cfg (nl-llm-weights-config wts))
                 (dim (plist-get cfg :dim))
                 (nlayers (plist-get cfg :layers))
                 (t0 (float-time))
                 (layers nil))
            (dotimes (ly nlayers)
              (push (nl-llm-wgpu-load-layer wts ly) layers)
              (when (= 0 (mod (1+ ly) 7))
                (nso-p2--say "  uploaded %d/%d layers, %.0fs" (1+ ly) nlayers
                             (- (float-time) t0))))
            (setq layers (nreverse layers))
            (nso-p2--say "resident load: %.0fs" (- (float-time) t0))
            (nso-p2--say "layer 0 against the CPU reference: rel %g"
                         (nso-encode-check-layer wts 0 (car layers) cfg))
            (unwind-protect
                (let ((rows nil) (i 0) (n (length todo)) (t1 (float-time)))
                  (dolist (it todo)
                    (let* ((ids (nl-llm-qwen-tok-encode tok (nth 2 it)))
                           (seq (length ids))
                           (x (make-vector (* seq dim) 0.0))
                           (mid nil) (p 0))
                      (dolist (tk ids)
                        (let ((row (nl-llm-weights-embed wts tk)))
                          (dotimes (j dim) (aset x (+ (* p dim) j) (aref row j))))
                        (setq p (1+ p)))
                      (let ((ly 0))
                        (dolist (lay layers)
                          (setq x (nso-encode-block lay x seq cfg))
                          (when (= ly nso-p2--mid-layer) (setq mid (copy-sequence x)))
                          (setq ly (1+ ly))))
                      ;; P1 measured mid depth as the better one, so only that
                      ;; is kept here; storing both again would double the file
                      ;; to re-answer a question already answered.
                      (push (list :kind (nth 0 it) :topic (nth 1 it) :text (nth 2 it)
                                  :seq seq
                                  :mid (let ((flat (nl-llm-wf-final-norm wts mid seq))
                                             (acc nil) (q 0))
                                         (while (< q seq)
                                           (let ((v (make-vector dim 0.0)))
                                             (dotimes (j dim)
                                               (aset v j (aref flat (+ (* q dim) j))))
                                             (push v acc))
                                           (setq q (1+ q)))
                                         (nreverse acc)))
                            rows)
                      (setq i (1+ i))
                      (when (= 0 (mod i 10))
                        (let ((el (- (float-time) t1)))
                          (nso-p2--say "encoded %d/%d, %.0fs elapsed, %.0fs remaining"
                                       i n el (* (/ el i) (- n i)))))
                      (when (= 0 (mod i nso-p2--checkpoint))
                        (nso-p2--write (append cached (reverse rows)) dim))))
                  (let ((all (append cached (nreverse rows))))
                    (nso-p2--say "states: %d rows" (length all))
                    (nso-p2--write all dim)
                    all))
              (dolist (lay layers) (nl-llm-wgpu-free-layer lay))))
        (nelisp-gpu-server-stop)))))

;;; --- the probe --------------------------------------------------------------

(defun nso-p2-probe (rows)
  (let* ((by-text (let ((h (make-hash-table :test 'equal)))
                    (dolist (r rows) (puthash (plist-get r :text) r h)) h))
         (raw-pool (lambda (r) (nso-pool-last (plist-get r :mid))))
         ;; Standardise, with statistics from the fitted rows only.
         ;;
         ;; Two measurements forced this.  The first P2 run fitted the head on
         ;; raw states at lr 0.5 and DIVERGED -- training accuracy 0.219 on an
         ;; eight-way task, loss 27.8 -- so its held-out numbers were an
         ;; optimiser artefact and said nothing about the architecture.  And
         ;; the states are badly anisotropic: pairwise cosine averages 0.923,
         ;; against 0.683 for the option embeddings, so they sit in a narrow
         ;; cone.  The Noul head survived that because a binary decision needs
         ;; one direction; separating eight does not.  Per-dimension
         ;; standardisation centres the cone away, and it is what every other
         ;; head in this repository has always been given.
         (std (nso-standardizer
               (let (o)
                 (dolist (r rows)
                   (when (or (and (eq (plist-get r :kind) 'option)
                                  ;; only the form actually in use
                                  (equal (plist-get r :text)
                                         (nso-p2--option-text (plist-get r :topic))))
                             (and (eq (plist-get r :kind) 'state)
                                  (not (nso-p2--held-p (plist-get r :topic)))))
                     (push (funcall raw-pool r) o)))
                 (nreverse o))))
         (pool (lambda (r) (nso-standardize std (funcall raw-pool r))))
         (opt (lambda (name)
                (funcall pool (gethash (nso-p2--option-text name) by-text))))
         (tmpl (plist-get nso-p2--data :template))
         (seen nil) (held nil))
    (nso-p2--say "option form: %s; states standardised from fitted rows only"
                 nso-p2--option-form)
    (dolist (tp (nso-p2--topics))
      (if (plist-get tp :held)
          (push (plist-get tp :name) held)
        (push (plist-get tp :name) seen)))
    (setq seen (nreverse seen) held (nreverse held))
    (nso-p2--say "topics: %d seen, %d held out" (length seen) (length held))
    (let* ((mk (lambda (names)
                 (let ((vecs (mapcar opt names)) (out nil))
                   (dolist (e (nso-p2--examples))
                     (when (member (plist-get e :topic) names)
                       (let ((r (gethash (format tmpl (plist-get e :text)) by-text))
                             (idx 0) (label nil) (i 0))
                         (dolist (nm names)
                           (when (equal nm (plist-get e :topic)) (setq label i))
                           (setq i (1+ i)))
                         (ignore idx)
                         (push (list :state (funcall pool r) :options vecs :label label)
                               out))))
                   (nreverse out))))
           (train (funcall mk seen))
           (freeze (and (getenv "NSO_P2_FREEZE_C") t))
           (model (nso-choice-train train 600 0.5 0.02 freeze))
           (acc-seen (nso-choice-accuracy model train))
           (held-only (funcall mk held))
           (acc-held (nso-choice-accuracy model held-only))
           (all-names (append seen held))
           (all-opts (mapcar opt all-names))
           ;; Held-out sentences scored against every topic, seen ones
           ;; included: the harder and more realistic setting.
           (held-vs-all (let ((out nil))
                          (dolist (e (nso-p2--examples))
                            (when (member (plist-get e :topic) held)
                              (let ((r (gethash (format tmpl (plist-get e :text)) by-text))
                                    (label nil) (i 0))
                                (dolist (nm all-names)
                                  (when (equal nm (plist-get e :topic)) (setq label i))
                                  (setq i (1+ i)))
                                (push (list :state (funcall pool r) :options all-opts
                                            :label label)
                                      out))))
                          (nreverse out)))
           (acc-all (nso-choice-accuracy model held-vs-all))
           (ci-held (nso-probe-wilson (round (* acc-held (length held-only)))
                                      (length held-only)))
           (ci-all (nso-probe-wilson (round (* acc-all (length held-vs-all)))
                                     (length held-vs-all))))
      (nso-p2--say "option-only direction: %s" (if freeze "FROZEN" "learned"))
      (nso-p2--say "seen topics (%d-way): train acc %.3f" (length seen) acc-seen)
      (nso-p2--say "HELD-OUT topics (%d-way): acc %.3f [%.3f,%.3f] chance %.3f"
                   (length held) acc-held (car ci-held) (cdr ci-held)
                   (/ 1.0 (length held)))
      (nso-p2--say "HELD-OUT vs all %d topics: acc %.3f [%.3f,%.3f] chance %.3f"
                   (length all-names) acc-all (car ci-all) (cdr ci-all)
                   (/ 1.0 (length all-names)))
      (with-temp-file nso-p2--results
        (insert "#+TITLE: P2 results -- Choice over options held out from training\n")
        (insert (format "#+DATE: %s\n\n" (format-time-string "%Y-%m-%d %H:%M")))
        (insert (format "%d topics, %d examples.  Fitted on %d topics; the other %d\n"
                        (length all-names) (length (nso-p2--examples))
                        (length seen) (length held)))
        (insert "were absent from training entirely, sentences and option text alike.\n\n")
        (insert "| evaluation | n | options | accuracy | 95% CI | chance |\n")
        (insert "|------------+---+---------+----------+--------+--------|\n")
        (insert (format "| seen topics (fit) | %d | %d | %.3f | -- | %.3f |\n"
                        (length train) (length seen) acc-seen (/ 1.0 (length seen))))
        (insert (format "| held-out topics | %d | %d | %.3f | [%.3f,%.3f] | %.3f |\n"
                        (length held-only) (length held) acc-held
                        (car ci-held) (cdr ci-held) (/ 1.0 (length held))))
        (insert (format "| held-out vs all | %d | %d | %.3f | [%.3f,%.3f] | %.3f |\n"
                        (length held-vs-all) (length all-names) acc-all
                        (car ci-all) (cdr ci-all) (/ 1.0 (length all-names))))
        (insert (format "\nHeld-out topics: %s\n" (mapconcat #'identity held ", "))))
      (nso-p2--say "report written to %s" nso-p2--results))))

;;; --- driver -----------------------------------------------------------------

(nso-p2--say "stage: %s" nso-p2--stage)
(condition-case err
    (cond
     ((equal nso-p2--stage "tokenize")
      (require 'nl-llm-qwen-tokenizer)
      (let ((tok (nl-llm-qwen-tok-load (expand-file-name "tokenizer.bin" nso-p2--donor)))
            (total 0) (mx 0))
        (dolist (it (nso-p2--items))
          (let ((n (length (nl-llm-qwen-tok-encode tok (nth 2 it)))))
            (setq total (+ total n) mx (max mx n))))
        (nso-p2--say "%d items, %d tokens total, max %d, est %.0f min"
                     (length (nso-p2--items)) total mx
                     (/ (* 10.0 (length (nso-p2--items))) 60.0))))
     ((equal nso-p2--stage "probe")
      (nso-p2-probe (nso-p2--cached)))
     (t (nso-p2-probe (nso-p2-encode))))
  (error (nso-p2--say "FAILED: %s" (error-message-string err))
         (signal (car err) (cdr err))))

(nso-p2--say "done")

;;; p2-choice.el ends here
