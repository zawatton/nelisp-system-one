;;; score-encode-probe.el --- Score: encode the commitment set, then judge it -*- lexical-binding: t; -*-

;; Same three stages as `p1-encode-probe.el', selected by NSO_SCORE_STAGE
;; (default "all"): tokenize, encode, probe.  The encode is the expensive one
;; and writes build/score-states.eld, so the probe can be re-run without
;; touching the GPU.
;;
;; The analysis is fixed by the pre-registration in section 6 of
;; `docs/design/01-system-one.org', committed before this file ran for the
;; first time.  One configuration -- mid layer, last-token pooling, what P1
;; shipped -- rather than the best of six, and both heads on identical
;; standardised features with identical steps, learning rate and L2.  Giving
;; two heads different optimiser budgets and comparing them measures the
;; budget; that has already cost this repository one result.
;;
;; The gates, in the order they are read:
;;
;;   0. convergence     both heads' final gradient norm below 0.05, else the
;;                      comparison is VOID and reported as void
;;   1. negative        shuffled training levels must land at chance and no
;;                      better than the constant answer
;;   2. ordinal claim   MAE(nominal) - MAE(ordinal) by paired bootstrap
;;   3. composition     hard-subset MAE below 1.200, the floor the dataset's
;;                      bag-identical pairing imposes on any bag of words
;;
;; Everything after that is reported and not gated.

(defvar nso-sc--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(defun nso-sc--sib (name) (expand-file-name (concat "../../" name) nso-sc--here))
(defun nso-sc--own (name) (expand-file-name (concat "../" name) nso-sc--here))

(add-to-list 'load-path (nso-sc--own "lisp"))
(add-to-list 'load-path (nso-sc--own "build/elc"))
(dolist (d '("nelisp-llm/lisp" "nelisp-photon/lisp" "nelisp-gpu/lisp"))
  (add-to-list 'load-path (nso-sc--sib d)))

;;; A stale sibling cache must not be able to win quietly
;;
;; build/elc holds byte-compiled copies of the sibling lisp and sits in FRONT
;; of the sibling sources on the load path, so an .elc older than its .el
;; shadows an edited file.  That happened on this phase's first encode: the
;; cache was fourteen hours behind nelisp-llm's weight loader, the layers
;; uploaded, the CPU reference check passed at 3.7e-09 -- and the first block
;; died with a wrong-type-argument naming a tensor, which looks like anything
;; except a load-path problem.
;;
;; The Makefile now rebuilds the cache as a dependency, which is the actual
;; fix.  This is the backstop, and it checks the quantity that matters
;; directly -- the mtime of the exact files on the load path -- rather than a
;; proxy for it.

(defun nso-sc--check-deps-fresh ()
  "Signal when any cached sibling .elc is older than the source it shadows."
  (let ((cache (nso-sc--own "build/elc"))
        (stale nil))
    (when (file-directory-p cache)
      (dolist (elc (directory-files cache t "\\.elc\\'"))
        (let ((base (file-name-base elc)))
          (dolist (d '("nelisp-llm/lisp" "nelisp-photon/lisp" "nelisp-gpu/lisp"))
            (let ((src (expand-file-name (concat base ".el") (nso-sc--sib d))))
              (when (and (file-readable-p src)
                         (time-less-p (file-attribute-modification-time
                                       (file-attributes elc))
                                      (file-attribute-modification-time
                                       (file-attributes src))))
                (push base stale)))))))
    (when stale
      (error (concat "score: %d cached .elc are older than their sources (%s). "
                     "build/elc shadows them on the load path.  Run `make deps'")
             (length stale) (mapconcat #'identity (sort stale #'string<) ", ")))))

(nso-sc--check-deps-fresh)

(require 'nso-score)
(require 'nso-probe)
(require 'nso-stub)

(defvar nso-sc--stage (or (getenv "NSO_SCORE_STAGE") "all"))
(defvar nso-sc--donor (nso-sc--sib "nelisp-llm/build/donor/qwen3-0.6b"))
(defvar nso-sc--build (nso-sc--own "build"))
(defvar nso-sc--states
  (or (getenv "NSO_SCORE_STATES")
      (expand-file-name "score-states.eld" nso-sc--build)))
(defvar nso-sc--results
  (or (getenv "NSO_SCORE_RESULTS")
      (expand-file-name "score-results.org" nso-sc--build)))
(defvar nso-sc--log (expand-file-name "score-progress.log" nso-sc--build))
(defvar nso-sc--mid-layer 13 "Zero-based mid-depth capture point, as P1 shipped.")

;; Fixed by the pre-registration.  Named constants rather than literals
;; scattered through the file, so that changing one is a visible edit.
(defvar nso-sc--steps 6000
  "A CAP on iterations, not a count.

The pre-registration named 600, and gate 0 voided two runs on it: the first
because a fixed step size overshot at dim 1024, the second because 600
iterations of a line search converged on synthetic ordinal data and left the
real features at a gradient norm of 0.141 against the gate's 0.05.

Raising it amends a committed protocol, so the reasoning belongs here rather
than in a commit message.  It is decided on a TRAIN-side quantity -- the
gradient norm, swept at 600/1200/2400/4800 before any held-out number was
looked at -- and it applies to BOTH heads under the same rule: each stops when
it is an order of magnitude inside the gate.  It cannot buy a favourable
comparison, because the nominal head is already at 5e-07 by iteration 600 and
its training loss does not move by a digit in the 4200 after that.  The only
head a larger cap can help is the ordinal one, which is the head gate 2 is
about.")
(defvar nso-sc--lr 0.5)
(defvar nso-sc--l2 0.01)
(defvar nso-sc--gnorm-limit 0.05)
(defvar nso-sc--bag-mae-floor 1.200)
(defvar nso-sc--bag-acc-ceiling 0.600)
(defvar nso-sc--boots 2000)
(defvar nso-sc--band-draws 400)

(defvar nso-sc--checkpoint-every
  (string-to-number (or (getenv "NSO_SCORE_CHECKPOINT") "20")))
(defvar nso-sc--progress-every
  (string-to-number (or (getenv "NSO_SCORE_PROGRESS") "10")))

(unless (file-directory-p nso-sc--build) (make-directory nso-sc--build t))

(defun nso-sc--say (fmt &rest args)
  (let ((line (concat (format-time-string "%H:%M:%S  ")
                      (apply #'format fmt args) "\n")))
    (princ line)
    (write-region line nil nso-sc--log t 'quiet)))

(defun nso-sc--rows-of (flat seq dim)
  (let ((out nil) (i 0))
    (while (< i seq)
      (let ((v (make-vector dim 0.0)))
        (dotimes (j dim) (aset v j (aref flat (+ (* i dim) j))))
        (push v out))
      (setq i (1+ i)))
    (nreverse out)))

;;; --- data -----------------------------------------------------------------

(defvar nso-sc--form
  (with-temp-buffer
    (insert-file-contents (nso-sc--own "data/score-commitment.eld"))
    (read (buffer-string))))
(defvar nso-sc--examples (append (plist-get nso-sc--form :examples) nil))
(defvar nso-sc--levels (append (plist-get nso-sc--form :levels) nil))
(defvar nso-sc--k (length nso-sc--levels))

;;; --- stage: tokenize ------------------------------------------------------

(defun nso-sc-tokenize ()
  (require 'nl-llm-qwen-tokenizer)
  (let* ((tok (nl-llm-qwen-tok-load (expand-file-name "tokenizer.bin" nso-sc--donor)))
         (tmpl (plist-get nso-sc--form :template))
         (out nil) (total 0) (mx 0) (mn 9999))
    (dolist (e nso-sc--examples)
      (let ((ids (nl-llm-qwen-tok-encode tok (format tmpl (plist-get e :text)))))
        (setq total (+ total (length ids))
              mx (max mx (length ids))
              mn (min mn (length ids)))
        (push (cons e ids) out)))
    (setq out (nreverse out))
    (let ((lim (getenv "NSO_SCORE_LIMIT")))
      (when lim
        ;; Truncation keeps whole scenarios: the file lists a scenario's
        ;; fifteen sentences together, and a smoke run that cuts one in half
        ;; would split a scenario across the train/held-out boundary, which is
        ;; the one thing the split exists to prevent.
        (let* ((n (string-to-number lim))
               (keep (* 15 (max 1 (/ n 15)))))
          (setq out (butlast out (max 0 (- (length out) keep))))
          (setq total 0)
          (dolist (p out) (setq total (+ total (length (cdr p)))))
          (nso-sc--say "NSO_SCORE_LIMIT: truncated to %d prompts (%d whole scenarios)"
                       (length out) (/ keep 15)))))
    (nso-sc--say "tokenised %d prompts: min %d, mean %.1f, max %d tokens"
                 (length out) mn (/ (float total) (length out)) mx)
    (nso-sc--say "estimated encode time at P1's measured rate: %.0f min"
                 (/ (* (/ 10.0 8.7) total) 60.0))
    out))

;;; --- stage: encode --------------------------------------------------------

(defun nso-sc--cached ()
  (when (and (file-readable-p nso-sc--states)
             (not (getenv "NSO_SCORE_FORCE_ENCODE")))
    (let ((saved (with-temp-buffer
                   (insert-file-contents nso-sc--states)
                   (read (buffer-string)))))
      (if (and (equal (plist-get saved :mid-layer) nso-sc--mid-layer)
               (integerp (plist-get saved :dim)))
          (plist-get saved :rows)
        (nso-sc--say "cache ignored: shape does not match this configuration")
        nil))))

(defun nso-sc--merge (cached new)
  (sort (append cached (copy-sequence new))
        (lambda (a b)
          (if (= (plist-get a :scenario) (plist-get b :scenario))
              (string< (plist-get a :text) (plist-get b :text))
            (< (plist-get a :scenario) (plist-get b :scenario))))))

(defun nso-sc--write-states (rows dim)
  (with-temp-file nso-sc--states
    (let ((print-level nil) (print-length nil))
      (prin1 (list :dim dim :mid-layer nso-sc--mid-layer :rows rows)
             (current-buffer)))))

(defun nso-sc-encode (tokenised)
  (require 'nl-llm-weights)
  (require 'nl-llm-weights-forward)
  (unless (require 'nl-llm-weights-gpu nil t)
    (error "score: nelisp-gpu is not loadable"))
  (require 'nso-encode)
  (nelisp-gpu-server-start)
  (unless (nelisp-gpu-server-up-p) (error "score: the GPU server would not start"))
  (unwind-protect
      (let* ((wts (nl-llm-weights-open (expand-file-name "weights.bin" nso-sc--donor)))
             (cfg (nl-llm-weights-config wts))
             (dim (plist-get cfg :dim))
             (nlayers (plist-get cfg :layers))
             (t0 (float-time))
             (cached (nso-sc--cached))
             (have (let ((h (make-hash-table :test 'equal)))
                     (dolist (r cached) (puthash (plist-get r :text) r h))
                     h))
             (todo (let (out)
                     (dolist (p tokenised)
                       (unless (gethash (plist-get (car p) :text) have) (push p out)))
                     (nreverse out)))
             (layers nil))
        (nso-sc--say "cache: %d rows reused, %d to encode (%.0f min)"
                     (length cached) (length todo) (/ (* 12.0 (length todo)) 60.0))
        (dotimes (ly (if todo nlayers 0))
          (push (nl-llm-wgpu-load-layer wts ly) layers)
          (when (= 0 (mod (1+ ly) 7))
            (nso-sc--say "  uploaded %d/%d layers, %.0fs elapsed"
                         (1+ ly) nlayers (- (float-time) t0))))
        (setq layers (nreverse layers))
        (nso-sc--say "resident load: %.0fs for %d layers" (- (float-time) t0) nlayers)
        (when todo
          (nso-sc--say "layer 0 against the CPU reference: rel %g"
                       (nso-encode-check-layer wts 0 (car layers) cfg)))
        (unwind-protect
            (let ((rows nil) (i 0) (n (length todo)) (t1 (float-time)))
              (dolist (pair todo)
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
                      (setq x (nso-encode-block lay x seq cfg))
                      (when (= ly nso-sc--mid-layer) (setq mid (copy-sequence x)))
                      (setq ly (1+ ly))))
                  (push (list :scenario (plist-get e :scenario)
                              :level (plist-get e :level)
                              :hard (plist-get e :hard)
                              :family (plist-get e :family)
                              :text (plist-get e :text)
                              :seq seq
                              :mid (nso-sc--rows-of (nl-llm-wf-final-norm wts mid seq)
                                                    seq dim)
                              :final (nso-sc--rows-of (nl-llm-wf-final-norm wts x seq)
                                                      seq dim))
                        rows)
                  (setq i (1+ i))
                  (when (= 0 (mod i nso-sc--progress-every))
                    (let ((el (- (float-time) t1)))
                      (nso-sc--say "encoded %d/%d, %.0fs elapsed, %.0fs remaining"
                                   i n el (* (/ el i) (- n i)))))
                  (when (= 0 (mod i nso-sc--checkpoint-every))
                    (nso-sc--write-states (nso-sc--merge cached (reverse rows)) dim)
                    (nso-sc--say "  checkpoint: %d rows on disk"
                                 (+ (length cached) (length rows))))))
              (setq rows (nso-sc--merge cached (nreverse rows)))
              (nso-sc--write-states rows dim)
              (nso-sc--say "states written: %d rows, %.0f MB" (length rows)
                           (/ (float (nth 7 (file-attributes nso-sc--states))) 1048576.0))
              rows)
          (dolist (lay layers) (nl-llm-wgpu-free-layer lay))))
    (nelisp-gpu-server-stop)))

;;; --- statistics -----------------------------------------------------------

(defun nso-sc--percentile (sorted q)
  (nth (min (1- (length sorted)) (max 0 (floor (* q (length sorted)))))
       sorted))

(defun nso-sc--boot-mean (xs rng draws)
  "Percentile bootstrap interval for the mean of XS.  Returns (LO MEAN HI)."
  (let* ((v (vconcat xs))
         (n (length v))
         (means nil)
         (sum 0.0))
    (dotimes (i n) (setq sum (+ sum (aref v i))))
    (dotimes (_ draws)
      (let ((s 0.0))
        (dotimes (_ n) (setq s (+ s (aref v (mod (nso-rng-next rng) n)))))
        (push (/ s n) means)))
    (setq means (sort means #'<))
    (list (nso-sc--percentile means 0.025)
          (/ sum n)
          (nso-sc--percentile means 0.975))))

(defun nso-sc--abs-errors (probs ys readout)
  (let ((out nil) (r ys))
    (dolist (p probs)
      (push (float (abs (- (funcall readout p) (car r)))) out)
      (setq r (cdr r)))
    (nreverse out)))

;;; --- the calibrated-by-construction band ---------------------------------
;;
;; P3's statistic, on the ordinal scale: the drop in NLL a recalibrator can
;; find, against what it finds on a model that needs no correction.  The band
;; is measured at the SAME split sizes the real run uses -- fitted on the
;; training count, evaluated on the held-out count -- because comparing a
;; statistic taken at one n against a floor taken at another is the error that
;; sank P3's ECE gate.

(defun nso-sc--draw-ordinal (rng n cuts sharpen)
  "N (margins . label) pairs from a model that is SHARPEN times overconfident."
  (let ((zs nil) (ys nil))
    (dotimes (_ n)
      (let* ((f (* 4.0 (- (nso-rng-float rng) 0.5)))
             (z (let ((v (make-vector (length cuts) 0.0)) (i 0))
                  (dolist (c cuts) (aset v i (- c f)) (setq i (1+ i)))
                  v))
             (p (nso-score-probs-from-margins z))
             (u (nso-rng-float rng))
             (acc 0.0) (lab (1- (length p))) (done nil))
        (dotimes (k (length p))
          (unless done
            (setq acc (+ acc (aref p k)))
            (when (>= acc u) (setq lab k done t))))
        (push (nso-score-scale-margins (list z) sharpen) zs)
        (push lab ys)))
    (cons (mapcar #'car (nreverse zs)) (nreverse ys))))

(defun nso-sc--band (nfit neval sharpen draws)
  "5th/95th percentile of the recalibration gain, at these split sizes."
  (let ((rng (nso-rng 24601)) (cuts '(-1.5 -0.5 0.5 1.5)) (gains nil))
    (dotimes (_ draws)
      (let* ((fit (nso-sc--draw-ordinal rng nfit cuts sharpen))
             (ev (nso-sc--draw-ordinal rng neval cuts sharpen))
             (temp (plist-get (nso-score-temperature-fit (car fit) (cdr fit))
                              :temperature)))
        (push (- (nso-score-temperature-nll (car ev) (cdr ev) 1.0)
                 (nso-score-temperature-nll (car ev) (cdr ev) temp))
              gains)))
    (setq gains (sort gains #'<))
    (list :p05 (nso-sc--percentile gains 0.05)
          :p95 (nso-sc--percentile gains 0.95))))

;;; --- stage: probe ---------------------------------------------------------

(defun nso-sc--fit (rows ys std k)
  (nso-score-train (mapcar (lambda (r) (nso-standardize std r)) rows)
                   ys k nso-sc--steps nso-sc--lr nso-sc--l2))

(defun nso-sc--line (label s)
  (if (plist-get s :empty)
      (format "  %-24s -- (n=0)" label)
    (format "  %-24s acc %.3f [%.3f,%.3f]  MAE %.3f  median %.3f  NLL %.3f  n=%d"
            label (plist-get s :accuracy)
            (plist-get s :ci-lo) (plist-get s :ci-hi)
            (plist-get s :mae) (plist-get s :mae-median)
            (plist-get s :nll) (plist-get s :n))))

(defun nso-sc-probe (rows)
  (let* ((pool (lambda (r) (nso-pool-last (plist-get r :mid))))
         (train nil) (test nil))
    (dolist (r rows)
      (if (= 0 (mod (plist-get r :scenario) 3)) (push r test) (push r train)))
    (setq train (nreverse train) test (nreverse test))
    (when (or (null train) (null test))
      (error "score: the split left one side empty (%d train, %d held-out)"
             (length train) (length test)))
    (let* ((k nso-sc--k)
           (try (mapcar (lambda (r) (plist-get r :level)) train))
           (tey (mapcar (lambda (r) (plist-get r :level)) test))
           (trx (mapcar pool train))
           (tex (mapcar pool test))
           (std (nso-standardizer trx))
           (ord (nso-sc--fit trx try std k))
           (nom (nso-score-nominal-train
                 (mapcar (lambda (v) (nso-standardize std v)) trx)
                 try k nso-sc--steps nso-sc--lr nso-sc--l2))
           (ord-p (mapcar (lambda (v) (nso-score-probs ord (nso-standardize std v))) tex))
           (nom-p (mapcar (lambda (v) (nso-score-nominal-probs
                                       nom (nso-standardize std v)))
                          tex))
           (ord-s (nso-score-report ord-p tey))
           (nom-s (nso-score-report nom-p tey))
           (base (nso-score-constant-mae tey k))
           (rng (nso-rng 90210))
           (report nil))
      (nso-sc--say "probe: %d train / %d held-out, %d levels, mid layer %d, last-token pool"
                   (length train) (length test) k nso-sc--mid-layer)
      (nso-sc--say "chance accuracy %.3f, best constant answer level %d at MAE %.3f"
                   (/ 1.0 k) (plist-get base :level) (plist-get base :mae))
      (nso-sc--say "")

      ;; --- gate 0: convergence -------------------------------------------
      (let* ((g-ord (plist-get ord :final-gnorm))
             (g-nom (plist-get nom :final-gnorm))
             (ok (and (< g-ord nso-sc--gnorm-limit) (< g-nom nso-sc--gnorm-limit))))
        (nso-sc--say "GATE 0 convergence: ordinal |g| %.2e, nominal |g| %.2e -- %s"
                     g-ord g-nom (if ok "PASS" "VOID"))
        (push (list :gate 0 :name "convergence" :pass ok
                    :detail (format "ordinal %.2e / nominal %.2e" g-ord g-nom))
              report)
        (unless ok
          (nso-sc--say "  the heads did not converge, so nothing below compares heads")))

      (nso-sc--say "%s" (nso-sc--line "ordinal, held-out" ord-s))
      (nso-sc--say "%s" (nso-sc--line "nominal, held-out" nom-s))

      ;; --- gate 1: negative control (mandatory) ---------------------------
      (let* ((shuffled (mapcar (lambda (_) (mod (nso-rng-next rng) k)) try))
             (ctl (nso-sc--fit trx shuffled std k))
             (ctl-s (nso-score-report
                     (mapcar (lambda (v) (nso-score-probs ctl (nso-standardize std v)))
                             tex)
                     tey))
             (ok (and (< (plist-get ctl-s :accuracy) (+ (/ 1.0 k) 0.15))
                      (> (plist-get ctl-s :mae) (* 0.85 (plist-get base :mae))))))
        (nso-sc--say "GATE 1 negative control (shuffled levels): acc %.3f, MAE %.3f -- %s"
                     (plist-get ctl-s :accuracy) (plist-get ctl-s :mae)
                     (if ok "PASS" "FAIL -- the pipeline scores without labels"))
        (push (list :gate 1 :name "negative control" :pass ok
                    :detail (format "acc %.3f against chance %.3f, MAE %.3f against %.3f"
                                    (plist-get ctl-s :accuracy) (/ 1.0 k)
                                    (plist-get ctl-s :mae) (plist-get base :mae)))
              report))

      ;; --- gate 2: does the ordering earn its keep? -----------------------
      (let* ((e-ord (nso-sc--abs-errors ord-p tey #'nso-score-mode))
             (e-nom (nso-sc--abs-errors nom-p tey #'nso-score-mode))
             (diff (let ((out nil) (a e-nom) (b e-ord))
                     (while a (push (- (car a) (car b)) out)
                            (setq a (cdr a) b (cdr b)))
                     (nreverse out)))
             (ci (nso-sc--boot-mean diff rng nso-sc--boots))
             (ok (> (nth 0 ci) 0.0)))
        (nso-sc--say "GATE 2 ordinal vs nominal: MAE difference %+.3f [%+.3f, %+.3f] -- %s"
                     (nth 1 ci) (nth 0 ci) (nth 2 ci)
                     (cond (ok "PASS, the ordering helps")
                           ((< (nth 2 ci) 0.0) "FAIL, the nominal head wins")
                           (t "UNSUPPORTED, the interval straddles zero")))
        (push (list :gate 2 :name "ordinal beats nominal" :pass ok
                    :detail (format "%+.3f [%+.3f, %+.3f]" (nth 1 ci) (nth 0 ci) (nth 2 ci)))
              report))

      ;; --- gate 3: composition, against a floor the data imposes ----------
      ;;
      ;; The pre-registration says "held-out MAE on the hard subset" and does
      ;; not say WHOSE.  That is an ambiguity in the rule, found while reading
      ;; the rule against a result, which is the worst moment to resolve one
      ;; quietly in a convenient direction.  So both heads are reported and
      ;; the verdict follows the head gate 2 selects -- declared here rather
      ;; than decided later.
      (let* ((split (lambda (probs pred)
                      (let ((ps nil) (ls nil) (rp probs) (rl tey))
                        (dolist (r test)
                          (when (funcall pred r) (push (car rp) ps) (push (car rl) ls))
                          (setq rp (cdr rp) rl (cdr rl)))
                        (cons (nreverse ps) (nreverse ls)))))
             (hard-p (lambda (r) (plist-get r :hard)))
             (easy-p (lambda (r) (not (plist-get r :hard))))
             (o-hard (funcall split ord-p hard-p))
             (o-easy (funcall split ord-p easy-p))
             (n-hard (funcall split nom-p hard-p))
             (n-easy (funcall split nom-p easy-p))
             (oh-s (nso-score-report (car o-hard) (cdr o-hard)))
             (oe-s (nso-score-report (car o-easy) (cdr o-easy)))
             (nh-s (nso-score-report (car n-hard) (cdr n-hard)))
             (ne-s (nso-score-report (car n-easy) (cdr n-easy)))
             (oh-ci (nso-sc--boot-mean
                     (nso-sc--abs-errors (car o-hard) (cdr o-hard) #'nso-score-mode)
                     rng nso-sc--boots))
             (nh-ci (nso-sc--boot-mean
                     (nso-sc--abs-errors (car n-hard) (cdr n-hard) #'nso-score-mode)
                     rng nso-sc--boots))
             ;; Which head the phase would ship, by gate 2's own comparison.
             (winner (if (< (plist-get ord-s :mae) (plist-get nom-s :mae))
                         'ordinal 'nominal))
             (ci (if (eq winner 'ordinal) oh-ci nh-ci))
             (ok (< (nth 2 ci) nso-sc--bag-mae-floor)))
        (nso-sc--say "%s" (nso-sc--line "  ordinal / hard" oh-s))
        (nso-sc--say "%s" (nso-sc--line "  ordinal / easy" oe-s))
        (nso-sc--say "%s" (nso-sc--line "  nominal / hard" nh-s))
        (nso-sc--say "%s" (nso-sc--line "  nominal / easy" ne-s))
        (nso-sc--say "  hard MAE with interval: ordinal %.3f [%.3f, %.3f], nominal %.3f [%.3f, %.3f]"
                     (nth 1 oh-ci) (nth 0 oh-ci) (nth 2 oh-ci)
                     (nth 1 nh-ci) (nth 0 nh-ci) (nth 2 nh-ci))
        (nso-sc--say "GATE 3 composition (on the %s head, which gate 2 selects):"
                     (symbol-name winner))
        (nso-sc--say "  MAE %.3f [%.3f, %.3f] against the %.3f bag floor -- %s"
                     (nth 1 ci) (nth 0 ci) (nth 2 ci) nso-sc--bag-mae-floor
                     (if ok "PASS" "FAIL"))
        (nso-sc--say "  secondary: hard accuracy ordinal %.3f / nominal %.3f against the %.3f ceiling"
                     (plist-get oh-s :accuracy) (plist-get nh-s :accuracy)
                     nso-sc--bag-acc-ceiling)
        (push (list :gate 3 :name (format "composition (%s head)" winner) :pass ok
                    :detail (format "MAE %.3f [%.3f, %.3f] vs floor %.3f; acc %.3f vs ceiling %.3f"
                                    (nth 1 ci) (nth 0 ci) (nth 2 ci) nso-sc--bag-mae-floor
                                    (if (eq winner 'ordinal)
                                        (plist-get oh-s :accuracy)
                                      (plist-get nh-s :accuracy))
                                    nso-sc--bag-acc-ceiling))
              report)
        (setq report (cons (list :subsets (list :ordinal-hard oh-s :ordinal-easy oe-s
                                                :nominal-hard nh-s :nominal-easy ne-s
                                                :winner winner))
                           report)))

      ;; --- reported, not gated --------------------------------------------
      (nso-sc--say "")
      (let* ((oof (nso-score-oof-margins
                   train try (mapcar (lambda (r) (plist-get r :scenario)) train)
                   (lambda (_a _b) pool) k 3 nso-sc--steps nso-sc--lr nso-sc--l2))
             (tfit (nso-score-temperature-fit oof try))
             (temp (plist-get tfit :temperature))
             (te-margins (mapcar (lambda (v) (nso-score-margins ord (nso-standardize std v)))
                                 tex))
             (nll-before (nso-score-temperature-nll te-margins tey 1.0))
             (nll-after (nso-score-temperature-nll te-margins tey temp))
             (gain (- nll-before nll-after))
             (band (nso-sc--band (length train) (length test) 1.0 nso-sc--band-draws))
             (over (nso-sc--band (length train) (length test) 0.25 nso-sc--band-draws))
             (after-p (mapcar #'nso-score-probs-from-margins
                              (nso-score-scale-margins te-margins temp)))
             (after-s (nso-score-report after-p tey))
             (inside (and (>= gain (plist-get band :p05))
                          (<= gain (plist-get band :p95)))))
        (nso-sc--say "calibration (reported, not a gate): T = %.3f%s, fitted out of fold"
                     temp (if (plist-get tfit :saturated) " AT BOUND" ""))
        (nso-sc--say "  NLL %.4f -> %.4f, gain %+.4f; calibrated band [%+.4f, %+.4f] -- %s"
                     nll-before nll-after gain
                     (plist-get band :p05) (plist-get band :p95)
                     (if inside "inside: no miscalibration visible at this n"
                       "outside: the raw output was not calibrated"))
        (nso-sc--say "  control, 4x overconfident: [%+.4f, %+.4f] -- %s"
                     (plist-get over :p05) (plist-get over :p95)
                     (if (> (plist-get over :p05) (plist-get band :p95))
                         "separated" "OVERLAPS, the statistic is blind here"))
        ;; An ordinal temperature can move the answer, unlike the binary one,
        ;; so accuracy and MAE are reported on both sides of it.
        (nso-sc--say "  accuracy %.3f -> %.3f, MAE %.3f -> %.3f across the temperature"
                     (plist-get ord-s :accuracy) (plist-get after-s :accuracy)
                     (plist-get ord-s :mae) (plist-get after-s :mae))
        (push (list :calibration (list :temp temp :gain gain :band band :over over
                                       :inside inside :after after-s))
              report))

      (nso-sc--say "  unimodal held-out distributions: %.1f%%"
                   (* 100 (plist-get ord-s :unimodal)))
      (nso-sc--say "  readout: MAE %.3f under the mode (the contract), %.3f under the median"
                   (plist-get ord-s :mae) (plist-get ord-s :mae-median))

      (setq report (nreverse report))
      (nso-sc--write-report report ord-s nom-s base (length train) (length test))
      report)))

(defun nso-sc--write-report (report ord-s nom-s base ntrain ntest)
  (with-temp-file nso-sc--results
    (insert "#+TITLE: Score results -- a cumulative-link head on a frozen donor\n")
    (insert (format "#+DATE: %s\n\n" (format-time-string "%Y-%m-%d %H:%M")))
    (insert (format "%d train / %d held-out, split by scenario.  Mid layer %d,\n"
                    ntrain ntest nso-sc--mid-layer))
    (insert "last-token pooling, both heads on identical standardised features\n")
    (insert (format "with %d steps at lr %.2f and L2 %.2f.  Chance %.3f, best\n"
                    nso-sc--steps nso-sc--lr nso-sc--l2 (/ 1.0 nso-sc--k)))
    (insert (format "constant answer %.3f MAE.\n\n" (plist-get base :mae)))
    (insert "| head | accuracy | 95% CI | MAE (mode) | MAE (median) | NLL |\n")
    (insert "|------+----------+--------+------------+--------------+-----|\n")
    (dolist (row (list (cons "ordinal" ord-s) (cons "nominal" nom-s)))
      (let ((s (cdr row)))
        (insert (format "| %s | %.3f | [%.3f,%.3f] | %.3f | %.3f | %.3f |\n"
                        (car row) (plist-get s :accuracy)
                        (plist-get s :ci-lo) (plist-get s :ci-hi)
                        (plist-get s :mae) (plist-get s :mae-median)
                        (plist-get s :nll)))))
    (insert "\n| gate | what it asks | verdict | detail |\n")
    (insert "|------+--------------+---------+--------|\n")
    (dolist (r report)
      (when (plist-get r :name)
        (insert (format "| %d | %s | %s | %s |\n"
                        (plist-get r :gate) (plist-get r :name)
                        (if (plist-get r :pass) "PASS" "FAIL")
                        (plist-get r :detail)))))
    (dolist (r report)
      (when (plist-get r :subsets)
        (let* ((x (plist-get r :subsets))
               (rows (list (list "ordinal / hard" (plist-get x :ordinal-hard))
                           (list "ordinal / easy" (plist-get x :ordinal-easy))
                           (list "nominal / hard" (plist-get x :nominal-hard))
                           (list "nominal / easy" (plist-get x :nominal-easy)))))
          (insert "\n| head / subset | accuracy | MAE | n | what a bag of words can do |\n")
          (insert "|---------------+----------+-----+---+----------------------------|\n")
          (dolist (row rows)
            (let ((s (cadr row)))
              (insert (format "| %s | %.3f | %.3f | %d | %s |\n"
                              (car row) (plist-get s :accuracy) (plist-get s :mae)
                              (plist-get s :n)
                              (if (string-match-p "hard" (car row))
                                  (format "acc <= %.3f, MAE >= %.3f"
                                          nso-sc--bag-acc-ceiling nso-sc--bag-mae-floor)
                                "1.000 and 0.000, measured")))))
          (insert (format "\nGate 3 is taken on the %s head, which gate 2 selects.\n"
                          (plist-get x :winner))))))
    (dolist (r report)
      (when (plist-get r :calibration)
        (let* ((c (plist-get r :calibration))
               (b (plist-get c :band)))
          (insert (format "\nCalibration (reported, not gated): T = %.3f out of fold,\n"
                          (plist-get c :temp)))
          (insert (format "gain %+.4f against a calibrated band of [%+.4f, %+.4f] -- %s.\n"
                          (plist-get c :gain) (plist-get b :p05) (plist-get b :p95)
                          (if (plist-get c :inside) "inside" "outside")))
          (insert (format "Across the temperature: accuracy %.3f -> %.3f, MAE %.3f -> %.3f.\n"
                          (plist-get ord-s :accuracy)
                          (plist-get (plist-get c :after) :accuracy)
                          (plist-get ord-s :mae)
                          (plist-get (plist-get c :after) :mae)))
          (insert "An ordinal temperature is not monotone in the answer, so those\n")
          (insert "two pairs of numbers are not guaranteed to be equal.\n"))))
    (insert (format "\nUnimodal held-out distributions: %.1f%%.\n"
                    (* 100 (plist-get ord-s :unimodal)))))
  (nso-sc--say "report written to %s" nso-sc--results))

;;; --- driver ---------------------------------------------------------------

(nso-sc--say "stage: %s" nso-sc--stage)
(condition-case err
    (cond
     ((equal nso-sc--stage "tokenize") (nso-sc-tokenize))
     ((equal nso-sc--stage "probe")
      (let ((saved (with-temp-buffer
                     (insert-file-contents nso-sc--states)
                     (read (buffer-string)))))
        (nso-sc-probe (plist-get saved :rows))))
     (t (nso-sc-probe (nso-sc-encode (nso-sc-tokenize)))))
  (error
   (nso-sc--say "FAILED: %s" (error-message-string err))
   (signal (car err) (cdr err))))

(nso-sc--say "done")

;;; score-encode-probe.el ends here
