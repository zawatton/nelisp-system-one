;;; p2-remedy.el --- the two P2 remedies, judged by the rule committed first -*- lexical-binding: t; -*-

;; Section 6's "P2 remedies", pre-registered before the mechanisms existed and
;; before anything here was run.  This file applies that rule; it does not
;; decide it.  Where the two disagree the document wins and this is the bug.
;;
;; Four heads on identical features with identical optimiser rules:
;;
;;   A  diagonal, per-axis standardise   -- the shipped baseline, 0.344 on train
;;   B  diagonal, whitening
;;   C  low rank r, per-axis standardise
;;   D  low rank r, whitening
;;
;; and four gates read in order: convergence, then a training-side screen, then
;; P2's own held-out-topic criterion, with a shuffled-label control read first
;; of all.
;;
;; Reads build/p2-states.eld.  No GPU, no donor.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/p2-remedy.el

(defvar nso-rm--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-rm--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-rm--here))

(require 'nso-choice)
(require 'nso-probe)
(require 'nso-stub)

(defvar nso-rm--states (expand-file-name "../build/p2-states.eld" nso-rm--here))
(defvar nso-rm--data (expand-file-name "../data/choice-intent.eld" nso-rm--here))
(defvar nso-rm--results (expand-file-name "../build/p2-remedy.org" nso-rm--here))

;; Fixed by the pre-registration.
(defvar nso-rm--ranks '(4 8 16))
(defvar nso-rm--steps 6000)
(defvar nso-rm--lr 0.5)
(defvar nso-rm--l2 0.02)
(defvar nso-rm--gnorm-limit 0.05)
(defvar nso-rm--shrink 0.1)
(defvar nso-rm--boots 2000)

(defvar nso-rm--log (expand-file-name "../build/p2-remedy-progress.log" nso-rm--here))

(defun nso-rm--say (fmt &rest args)
  "Print to stdout AND append to the progress file.

Both, because batch Emacs buffers stdout: a run of this length shows nothing
at all until it exits, and if it dies it shows nothing ever.  The P1 encoder
in this repository says exactly that at the top of its own file, and this one
was written without it anyway -- fifty-nine minutes in, with no way to tell
whether it was a tenth done or nine tenths.  `write-region' appends
immediately, so this survives what stdout does not."
  (let ((line (concat (format-time-string "%H:%M:%S  ")
                      (apply #'format fmt args) "\n")))
    (princ line)
    (write-region line nil nso-rm--log t 'quiet)))

;;; --- data ----------------------------------------------------------------

(defvar nso-rm--form
  (with-temp-buffer (insert-file-contents nso-rm--data) (read (buffer-string))))
(defvar nso-rm--topics (append (plist-get nso-rm--form :topics) nil))
(defvar nso-rm--examples (append (plist-get nso-rm--form :examples) nil))
(defvar nso-rm--template (plist-get nso-rm--form :template))

(defun nso-rm--topic (name)
  (let (r) (dolist (tp nso-rm--topics) (when (equal name (plist-get tp :name)) (setq r tp))) r))
(defun nso-rm--held-p (name) (plist-get (nso-rm--topic name) :held))
(defun nso-rm--option-text (name) (plist-get (nso-rm--topic name) :description))

;;; --- geometry, reported ---------------------------------------------------

(defun nso-rm--mean-cosine (vs)
  "Mean pairwise cosine of VS.  §10.4 blames a cone at 0.923."
  (let ((s 0.0) (k 0) (n (length vs)))
    (dotimes (i n)
      (dotimes (j i)
        (let* ((a (nth i vs)) (b (nth j vs))
               (na (sqrt (nso-dot a a))) (nb (sqrt (nso-dot b b))))
          (setq s (+ s (/ (nso-dot a b) (max 1.0e-12 (* na nb)))) k (1+ k)))))
    (/ s (max 1 k))))

;;; --- statistics ------------------------------------------------------------

(defun nso-rm--wilson (k n)
  (nso-wilson k n))

(defun nso-rm--boot-mean (xs rng draws)
  "Percentile bootstrap interval for the mean of XS.  Returns (LO MEAN HI)."
  (let* ((v (vconcat xs)) (n (length v)) (means nil) (sum 0.0))
    (dotimes (i n) (setq sum (+ sum (aref v i))))
    (dotimes (_ draws)
      (let ((s 0.0))
        (dotimes (_ n) (setq s (+ s (aref v (mod (nso-rng-next rng) n)))))
        (push (/ s n) means)))
    (setq means (sort means #'<))
    (list (nth (floor (* 0.025 draws)) means)
          (/ sum n)
          (nth (min (1- draws) (floor (* 0.975 draws))) means))))

;;; --- the four heads, behind one interface ---------------------------------
;;
;; So that gate 0's convergence check and gate 1's screen can be applied
;; without knowing which head they hold.  A comparison that treats its arms
;; differently measures the treatment.

(defun nso-rm--fit (kind rank examples)
  (if (eq kind 'diagonal)
      (let ((m (nso-choice-train examples nso-rm--steps nso-rm--lr nso-rm--l2)))
        (list :model m :kind kind
              :gnorm (or (plist-get m :final-gnorm) 0.0)
              :params (* 2 (length (plist-get m :a)))))
    (let ((m (nso-choice-lowrank-train examples rank nso-rm--steps nso-rm--lr nso-rm--l2)))
      (list :model m :kind kind
            :gnorm (or (plist-get m :final-gnorm) 0.0)
            :params (* rank (length (plist-get (car examples) :state)))))))

(defun nso-rm--correct-p (fit e)
  (let* ((m (plist-get fit :model))
         (p (if (eq (plist-get fit :kind) 'diagonal)
                (nso-choice-probs m (plist-get e :state) (plist-get e :options))
              (nso-choice-lowrank-probs m (plist-get e :state) (plist-get e :options))))
         (best 0))
    (dotimes (j (length p)) (when (> (aref p j) (aref p best)) (setq best j)))
    (if (= best (plist-get e :label)) 1.0 0.0)))

(defun nso-rm--hits (fit examples)
  (mapcar (lambda (e) (nso-rm--correct-p fit e)) examples))

;;; --- building examples under a chosen feature transform --------------------

(defun nso-rm--builder (rows transform)
  "Return (POOL . OPT) closures for TRANSFORM, fitted on non-held rows only."
  (let* ((by-text (let ((h (make-hash-table :test 'equal)))
                    (dolist (r rows) (puthash (plist-get r :text) r h)) h))
         (raw (lambda (r) (nso-pool-last (plist-get r :mid))))
         (fit-rows (let (o)
                     (dolist (r rows)
                       (when (or (and (eq (plist-get r :kind) 'option)
                                      (equal (plist-get r :text)
                                             (nso-rm--option-text (plist-get r :topic))))
                                 (and (eq (plist-get r :kind) 'state)
                                      (not (nso-rm--held-p (plist-get r :topic)))))
                         (push (funcall raw r) o)))
                     (nreverse o)))
         (apply-fn
          (if (eq transform 'whiten)
              (let ((w (nso-whitener fit-rows nso-rm--shrink)))
                (lambda (v) (nso-whiten w v)))
            (let ((std (nso-standardizer fit-rows)))
              (lambda (v) (nso-standardize std v)))))
         (pool (lambda (r) (funcall apply-fn (funcall raw r)))))
    (list :pool pool
          :opt (lambda (name) (funcall pool (gethash (nso-rm--option-text name) by-text)))
          :by-text by-text
          :fit-raw fit-rows
          :fit-mapped (mapcar apply-fn fit-rows))))

(defun nso-rm--examples-for (b names)
  "Examples whose options are exactly NAMES, in that order."
  (let* ((opt (plist-get b :opt))
         (pool (plist-get b :pool))
         (by-text (plist-get b :by-text))
         (vecs (mapcar opt names))
         (out nil))
    (dolist (e nso-rm--examples)
      (when (member (plist-get e :topic) names)
        (let ((r (gethash (format nso-rm--template (plist-get e :text)) by-text))
              (label nil) (i 0))
          (dolist (nm names)
            (when (equal nm (plist-get e :topic)) (setq label i))
            (setq i (1+ i)))
          (push (list :state (funcall pool r) :options vecs :label label
                      :topic (plist-get e :topic))
                out))))
    (nreverse out)))

;;; --- the run ---------------------------------------------------------------

(let* ((saved (with-temp-buffer (insert-file-contents nso-rm--states)
                                (read (buffer-string))))
       (rows (plist-get saved :rows))
       (seen nil) (held nil))
  (dolist (tp nso-rm--topics)
    (if (plist-get tp :held) (push (plist-get tp :name) held)
      (push (plist-get tp :name) seen)))
  (setq seen (nreverse seen) held (nreverse held))
  (nso-rm--say "P2 remedies -- %d seen topics, %d held out\n" (length seen) (length held))

  ;; Both builders, once, in the scope that spans every user of them.  They
  ;; were built three times over: once for the geometry report, once inside
  ;; every `loo' call, and once per arm -- and four of those refit the
  ;; whitener's Jacobi sweep over a 100x100 Gram matrix.  Nothing about a
  ;; builder depends on which head, which rank or which fold is about to use
  ;; it.
  (let* ((rng (nso-rng 51515))
         (arms nil)
         (builders (list (cons 'standardise (nso-rm--builder rows 'standardise))
                         (cons 'whiten (nso-rm--builder rows 'whiten)))))
    ;; --- geometry, reported --------------------------------------------
    (let ((bs (cdr (assq 'standardise builders)))
          (bw (cdr (assq 'whiten builders))))
      (nso-rm--say "geometry of the fitting rows (mean pairwise cosine):")
      (nso-rm--say "  raw %.3f | standardised %.3f | whitened %.3f\n"
                   (nso-rm--mean-cosine (plist-get bs :fit-raw))
                   (nso-rm--mean-cosine (plist-get bs :fit-mapped))
                   (nso-rm--mean-cosine (plist-get bw :fit-mapped))))

    ;; --- rank selection, on the SEEN topics only -------------------------
    ;;
    ;; Leave one seen topic out, fit on the other seven, score the eighth over
    ;; the same eight options.  Train accuracy would pick the largest rank by
    ;; construction, so it is not what selects.
    (nso-rm--say "rank selection, leave-one-topic-out over the seen topics:")
    (nso-rm--say "  (nothing here touches a held-out topic)\n")
    (let ((loo (lambda (transform kind rank)
                 (let* ((b (cdr (assq transform builders)))
                        ;; And the examples are built once per sweep rather
                        ;; than once per fold: they do not depend on which
                        ;; topic is held out, only on which are in play.
                        (all (nso-rm--examples-for b seen))
                        (hits nil))
                   (nso-rm--say "    LOO %s/%s rank %s: %d folds"
                                transform kind rank (length seen))
                   (dolist (out-topic seen)
                     (let* ((tr (let (o) (dolist (e all)
                                           (unless (equal (plist-get e :topic) out-topic)
                                             (push e o)))
                                     (nreverse o)))
                            (te (let (o) (dolist (e all)
                                           (when (equal (plist-get e :topic) out-topic)
                                             (push e o)))
                                     (nreverse o)))
                            (t0 (float-time))
                            (fit (nso-rm--fit kind rank tr)))
                       (nso-rm--say "      fold %-18s %5.1fs  |g| %.2e"
                                    out-topic (- (float-time) t0)
                                    (plist-get fit :gnorm))
                       (setq hits (append hits (nso-rm--hits fit te)))))
                   hits))))
      (let ((a-hits (funcall loo 'standardise 'diagonal 0)))
        (nso-rm--say "  A  diagonal / standardise        LOO %.3f"
                     (/ (apply #'+ a-hits) (float (length a-hits))))
        (let ((b-hits (funcall loo 'whiten 'diagonal 0))
              (best-c nil) (best-c-rank nil) (best-d nil) (best-d-rank nil))
          (nso-rm--say "  B  diagonal / whiten             LOO %.3f"
                       (/ (apply #'+ b-hits) (float (length b-hits))))
          (dolist (r nso-rm--ranks)
            (let* ((ch (funcall loo 'standardise 'lowrank r))
                   (dh (funcall loo 'whiten 'lowrank r))
                   (ca (/ (apply #'+ ch) (float (length ch))))
                   (da (/ (apply #'+ dh) (float (length dh)))))
              (nso-rm--say "  C  low rank %2d / standardise     LOO %.3f" r ca)
              (nso-rm--say "  D  low rank %2d / whiten          LOO %.3f" r da)
              (when (or (null best-c) (> ca (/ (apply #'+ best-c) (float (length best-c)))))
                (setq best-c ch best-c-rank r))
              (when (or (null best-d) (> da (/ (apply #'+ best-d) (float (length best-d)))))
                (setq best-d dh best-d-rank r))))
          (nso-rm--say "\n  selected: C at rank %d, D at rank %d" best-c-rank best-d-rank)
          (setq arms (list (list :name "A diagonal/standardise" :kind 'diagonal :rank 0
                                 :transform 'standardise :loo a-hits)
                           (list :name "B diagonal/whiten" :kind 'diagonal :rank 0
                                 :transform 'whiten :loo b-hits)
                           (list :name (format "C lowrank-%d/standardise" best-c-rank)
                                 :kind 'lowrank :rank best-c-rank
                                 :transform 'standardise :loo best-c)
                           (list :name (format "D lowrank-%d/whiten" best-d-rank)
                                 :kind 'lowrank :rank best-d-rank
                                 :transform 'whiten :loo best-d))))))

    ;; --- fit each arm on all seen topics ---------------------------------
    (dolist (arm arms)
      (let* ((b (cdr (assq (plist-get arm :transform) builders)))
             (tr (nso-rm--examples-for b seen))
             (fit (nso-rm--fit (plist-get arm :kind) (plist-get arm :rank) tr)))
        (plist-put arm :fit fit)
        (plist-put arm :builder b)
        (plist-put arm :train (/ (apply #'+ (nso-rm--hits fit tr)) (float (length tr))))))

    ;; --- gate 3 first: the control ---------------------------------------
    (nso-rm--say "\nGATE 3 negative control (shuffled labels), read first:")
    (let ((ok t))
      (dolist (arm arms)
        (let* ((b (plist-get arm :builder))
               (tr (nso-rm--examples-for b seen))
               (n (length (plist-get (car tr) :options)))
               (shuffled (mapcar (lambda (e)
                                   (let ((c (copy-sequence e)))
                                     (plist-put c :label (mod (nso-rng-next rng) n))))
                                 tr))
               (fit (nso-rm--fit (plist-get arm :kind) (plist-get arm :rank) shuffled))
               (te (nso-rm--examples-for b held))
               (acc (/ (apply #'+ (nso-rm--hits fit te)) (float (length te)))))
          (nso-rm--say "  %-28s held-out acc %.3f (chance %.3f)"
                       (plist-get arm :name) acc (/ 1.0 (length held)))
          (when (> acc (+ (/ 1.0 (length held)) 0.20)) (setq ok nil))))
      (nso-rm--say "  -- %s" (if ok "PASS" "FAIL, nothing below is read"))
      (unless ok (error "p2-remedy: the control scored without labels")))

    ;; --- gate 0: convergence ---------------------------------------------
    (nso-rm--say "\nGATE 0 convergence:")
    (let ((ok t))
      (dolist (arm arms)
        (let ((g (plist-get (plist-get arm :fit) :gnorm)))
          (nso-rm--say "  %-28s |g| %.2e  params %d  train %.3f"
                       (plist-get arm :name) g
                       (plist-get (plist-get arm :fit) :params)
                       (plist-get arm :train))
          (when (>= g nso-rm--gnorm-limit) (setq ok nil))))
      (nso-rm--say "  -- %s" (if ok "PASS" "VOID"))
      (unless ok (error "p2-remedy: an arm did not converge")))

    ;; --- gate 1: the training screen --------------------------------------
    (nso-rm--say "\nGATE 1 training screen -- leave-one-topic-out against A:")
    (let ((a-loo (plist-get (car arms) :loo)))
      (dolist (arm (cdr arms))
        (let* ((d (let ((out nil) (x (plist-get arm :loo)) (y a-loo))
                    (while x (push (- (car x) (car y)) out)
                           (setq x (cdr x) y (cdr y)))
                    (nreverse out)))
               (ci (nso-rm--boot-mean d rng nso-rm--boots))
               (pass (> (nth 0 ci) 0.0)))
          (plist-put arm :screened pass)
          (nso-rm--say "  %-28s %+.3f [%+.3f, %+.3f] -- %s"
                       (plist-get arm :name) (nth 1 ci) (nth 0 ci) (nth 2 ci)
                       (cond (pass "PASS, carried to held-out")
                             ((< (nth 2 ci) 0.0) "worse than A")
                             (t "not separated from A"))))))

    ;; --- gate 2: P2's own criterion ---------------------------------------
    (nso-rm--say "\nGATE 2 held-out topics (%d options, chance %.3f):"
                 (length held) (/ 1.0 (length held)))
    (dolist (arm arms)
      (let* ((b (plist-get arm :builder))
             (te (nso-rm--examples-for b held))
             (hits (nso-rm--hits (plist-get arm :fit) te))
             (k (round (apply #'+ hits)))
             (n (length hits))
             (ci (nso-rm--wilson k n))
             (acc (/ (float k) n))
             (pass (> (car ci) (/ 1.0 (length held))))
             (screened (or (eq arm (car arms)) (plist-get arm :screened))))
        (nso-rm--say "  %-28s acc %.3f [%.3f, %.3f] n=%d -- %s%s"
                     (plist-get arm :name) acc (car ci) (cdr ci) n
                     (cond (pass "PASS")
                           ((>= acc 0.25) "inside the uncallable band")
                           (t "at or below chance"))
                     (if screened "" "  (did not pass gate 1)"))))

    ;; --- reported, not gated ----------------------------------------------
    (nso-rm--say "\nreported, not gated -- all twelve options:")
    (dolist (arm arms)
      (let* ((b (plist-get arm :builder))
             (all (nso-rm--examples-for b (append seen held)))
             (hits (nso-rm--hits (plist-get arm :fit) all)))
        (nso-rm--say "  %-28s acc %.3f n=%d (chance %.3f)"
                     (plist-get arm :name)
                     (/ (apply #'+ hits) (float (length hits))) (length hits)
                     (/ 1.0 (length (append seen held))))))

    (with-temp-file nso-rm--results
      (insert "#+TITLE: P2 remedies -- capacity and whitening\n")
      (insert (format "#+DATE: %s\n\n" (format-time-string "%Y-%m-%d %H:%M")))
      (insert "| head | params | train | LOO | held-out topics | 95% CI |\n")
      (insert "|------+--------+-------+-----+-----------------+--------|\n")
      (dolist (arm arms)
        (let* ((b (plist-get arm :builder))
               (te (nso-rm--examples-for b held))
               (hits (nso-rm--hits (plist-get arm :fit) te))
               (k (round (apply #'+ hits))) (n (length hits))
               (ci (nso-rm--wilson k n))
               (loo (plist-get arm :loo)))
          (insert (format "| %s | %d | %.3f | %.3f | %.3f | [%.3f,%.3f] |\n"
                          (plist-get arm :name)
                          (plist-get (plist-get arm :fit) :params)
                          (plist-get arm :train)
                          (/ (apply #'+ loo) (float (length loo)))
                          (/ (float k) n) (car ci) (cdr ci)))))
      (insert "\nChance is 0.250 on four held-out topics.  A Wilson lower bound\n")
      (insert "clears it at about 0.40 on 48 examples, so anything between is\n")
      (insert "uncallable here -- declared in section 6 before this ran.\n"))
    (nso-rm--say "\nreport written to %s" nso-rm--results)))

;;; p2-remedy.el ends here
