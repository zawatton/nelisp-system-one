;;; choice-test.el --- the Choice head, and the property it exists for -*- lexical-binding: t; -*-

;;; Commentary:

;; The acceptance criterion for P2 names one thing: Choice has to work on an
;; option set held out from training.  That is the whole argument of section
;; 3.1 -- a head that classifies the state into N fixed slots cannot be asked
;; a question whose options it has not met, and the option-scoring form can.
;;
;; So the suite here is built around a synthetic task where that property is
;; separable from everything else: options are directions in feature space, a
;; state is a noisy copy of its own option's direction, and the answer is
;; whichever option the state points at.  A head that has learned "compare the
;; state to the option" transfers to new directions.  One that has memorised
;; which slot won does not, and the second test below is what tells them
;; apart -- it evaluates on options drawn from directions never used in
;; training.
;;
;; Runs on synthetic features, so it needs no donor and no GPU.

;;; Code:

(require 'nso-choice)
(require 'nso-types)
(require 'nso-stub)
(load (expand-file-name "nso-test-helper.el"
                        (file-name-directory (or load-file-name buffer-file-name))))

(message "== choice ==")

(defvar ct--dim 24)

(defun ct--unit (rng dim)
  "A random direction."
  (let ((v (make-vector dim 0.0)) (s 0.0))
    (dotimes (j dim) (aset v j (- (nso-rng-float rng) 0.5)))
    (dotimes (j dim) (setq s (+ s (* (aref v j) (aref v j)))))
    (setq s (sqrt s))
    (dotimes (j dim) (aset v j (/ (aref v j) s)))
    v))

(defun ct--example (rng options label noise)
  "A state that points at OPTIONS' LABEL-th direction, plus NOISE."
  (let* ((o (nth label options))
         (s (make-vector ct--dim 0.0)))
    (dotimes (j ct--dim)
      (aset s j (+ (aref o j) (* noise (- (nso-rng-float rng) 0.5)))))
    (list :state s :options options :label label)))

(defun ct--set (rng options n noise)
  (let (out)
    (dotimes (i n)
      (push (ct--example rng options (mod i (length options)) noise) out))
    (nreverse out)))

;;; --- gradients against finite differences --------------------------------

(let* ((rng (nso-rng 7))
       (options (let (o) (dotimes (_ 4) (push (ct--unit rng ct--dim) o)) o))
       (examples (ct--set rng options 9 0.6))
       (model (list :a (let ((v (make-vector ct--dim 0.0)))
                         (dotimes (j ct--dim) (aset v j (+ 0.8 (* 0.4 (nso-rng-float rng)))))
                         v)
                    :c (let ((v (make-vector ct--dim 0.0)))
                         (dotimes (j ct--dim) (aset v j (- (nso-rng-float rng) 0.5)))
                         v)))
       (l2 0.03)
       (g (nso-choice-grad model examples l2))
       (eps 1.0e-5))
  (dolist (probe (list (cons :a :da) (cons :c :dc)))
    (let ((worst 0.0)
          (vec (plist-get model (car probe)))
          (want (plist-get g (cdr probe))))
      (dotimes (j ct--dim)
        (let ((orig (aref vec j)))
          (aset vec j (+ orig eps))
          (let ((lp (nso-choice-loss model examples l2)))
            (aset vec j (- orig eps))
            (let ((lm (nso-choice-loss model examples l2)))
              (aset vec j orig)
              (setq worst (max worst
                               (/ (abs (- (aref want j) (/ (- lp lm) (* 2 eps))))
                                  (max 1.0e-8 (abs (aref want j))))))))))
      (nso-t-lt (format "Choice dL/d%s matches finite differences"
                        (substring (symbol-name (car probe)) 1))
                worst 1.0e-5))))

;;; --- it learns the seen option set ---------------------------------------

(let* ((rng (nso-rng 11))
       (train-opts (let (o) (dotimes (_ 5) (push (ct--unit rng ct--dim) o)) o))
       (train (ct--set rng train-opts 100 0.9))
       (model (nso-choice-train train 400 0.5 0.02))
       (acc (nso-choice-accuracy model train)))
  (message "  seen options: train accuracy %.3f (chance %.3f)" acc 0.2)
  (nso-t-gt "the Choice head learns a seen option set" acc 0.7)

  ;; --- and transfers to an option set it has never seen -------------------
  ;;
  ;; The point of the whole design.  Fresh directions, fresh states, the same
  ;; fitted head: nothing about these options was available during training.
  (let* ((new-opts (let (o) (dotimes (_ 5) (push (ct--unit rng ct--dim) o)) o))
         (test (ct--set rng new-opts 100 0.9))
         (acc2 (nso-choice-accuracy model test)))
    (message "  UNSEEN options: accuracy %.3f (chance %.3f)" acc2 0.2)
    (nso-t-gt "and transfers to options never seen in training" acc2 0.6)

    ;; A different cardinality too, since the head must not depend on N.
    (let* ((wide-opts (let (o) (dotimes (_ 12) (push (ct--unit rng ct--dim) o)) o))
           (wide (ct--set rng wide-opts 120 0.9))
           (acc3 (nso-choice-accuracy model wide)))
      (message "  UNSEEN options, 12-way: accuracy %.3f (chance %.3f)" acc3 (/ 1.0 12))
      (nso-t-gt "and to a different number of them" acc3 0.4)))

  ;; Negative control: states unrelated to their options.  A head that scores
  ;; high here is reading something other than the state-option relation, and
  ;; every number above would be suspect.
  (let* ((noise-opts (let (o) (dotimes (_ 5) (push (ct--unit rng ct--dim) o)) o))
         (noise (let (out)
                  (dotimes (i 100)
                    (push (list :state (ct--unit rng ct--dim)
                                :options noise-opts
                                :label (mod i 5))
                          out))
                  (nreverse out)))
         (acc4 (nso-choice-accuracy model noise)))
    (message "  unrelated states: accuracy %.3f (chance %.3f)" acc4 0.2)
    (nso-t-lt "but sits at chance when the state says nothing about the option"
              acc4 0.35)))

;;; --- the answers are well typed ------------------------------------------

(let* ((rng (nso-rng 13))
       (names '(alpha beta gamma))
       (vecs (let (o) (dotimes (_ 3) (push (ct--unit rng ct--dim) o)) o))
       (train (ct--set rng vecs 60 0.8))
       (model (nso-choice-train train 200 0.5 0.02))
       (q (nso-make-choice "which direction?" names))
       (a (nso-choice-answer model (plist-get (car train) :state) names vecs)))
  (nso-t-green "a Choice answer passes the type gate" (nso-type-gate q a))
  (nso-t-num "its probabilities sum to one"
             (let ((s 0.0))
               (dolist (cell (plist-get a :probabilities)) (setq s (+ s (cdr cell))))
               s)
             1.0 1e-9)
  (nso-t "and the reported choice is the argmax of its own distribution"
         (let ((best nil) (mx -1.0))
           (dolist (cell (plist-get a :probabilities))
             (when (> (cdr cell) mx) (setq mx (cdr cell) best (car cell))))
           (eq best (plist-get a :choice)))))


;;; --- at the scale and geometry of real features --------------------------
;;
;; Everything above uses unit-norm directions with small components.  The real
;; states from a frozen donor do not look like that: RMS about 4, and pairwise
;; cosine averaging 0.923 -- they sit in a narrow cone, while their option
;; embeddings average 0.683.  P2 fitted this head on those and the shipped
;; setting DIVERGED (training accuracy 0.219 on an eight-way task, loss 27.8),
;; which the suite above could not have caught: at unit norm the same learning
;; rate is fine.
;;
;; This is the second time a synthetic suite has passed while the real scale
;; broke the optimiser -- the attention pool in P1 was the first -- so the
;; scale belongs in the suite rather than in a postmortem.  These checks do
;; not assert that the head succeeds; they assert what is actually true, which
;; is that raw features at this scale diverge and standardised ones do not.

(let* ((rng (nso-rng 29))
       (dim ct--dim)
       ;; A shared direction every state leans on, which is what produces the
       ;; anisotropy, plus per-example variation on top.
       (common (ct--unit rng dim))
       (scale 4.0)
       (opts (let (o) (dotimes (_ 5) (push (ct--unit rng dim) o)) o))
       (raw (let (out)
              (dotimes (i 80)
                (let* ((label (mod i 5))
                       (o (nth label opts))
                       (v (make-vector dim 0.0)))
                  (dotimes (j dim)
                    (aset v j (* scale (+ (* 0.9 (aref common j))
                                          (* 0.25 (aref o j))
                                          (* 0.1 (- (nso-rng-float rng) 0.5))))))
                  (push (list :state v :options opts :label label) out)))
              (nreverse out)))
       (cos (lambda (a b) (/ (nso-dot a b)
                             (* (sqrt (nso-dot a a)) (sqrt (nso-dot b b))))))
       (mean-cos (let ((sum 0.0) (k 0) (vs (mapcar (lambda (e) (plist-get e :state))
                                                   raw)))
                   (dotimes (i (length vs))
                     (dotimes (j (length vs))
                       (when (< i j)
                         (setq sum (+ sum (funcall cos (nth i vs) (nth j vs)))
                               k (1+ k)))))
                   (/ sum k)))
       (std (nso-standardizer (mapcar (lambda (e) (plist-get e :state)) raw)))
       (stdized (mapcar (lambda (e)
                          (list :state (nso-standardize std (plist-get e :state))
                                :options (plist-get e :options)
                                :label (plist-get e :label)))
                        raw))
       (m-raw (nso-choice-train raw 400 0.5 0.02))
       (m-std (nso-choice-train stdized 400 0.5 0.02))
       (l-raw (nso-choice-loss m-raw raw 0.02))
       (l-std (nso-choice-loss m-std stdized 0.02)))
  (message "  realistic geometry: mean pairwise cosine %.3f (real states: 0.923)"
           mean-cos)
  (message "  raw          train acc %.3f  loss %.4g"
           (nso-choice-accuracy m-raw raw) l-raw)
  (message "  standardised train acc %.3f  loss %.4g"
           (nso-choice-accuracy m-std stdized) l-std)
  (nso-t-gt "the synthetic states are anisotropic like the real ones"
            mean-cos 0.7)
  (nso-t-gt "raw features at this scale cost a real part of the fit"
            (- (nso-choice-accuracy m-std stdized)
               (nso-choice-accuracy m-raw raw))
            0.3)
  (nso-t-lt "and a real part of the loss" l-std (* 0.8 l-raw))
  ;; Written after measuring rather than before: the real data DIVERGED on raw
  ;; features (loss 27.8) and this synthetic does not go that far, it only
  ;; loses half the fit.  So the assertion is the effect that reproduces, not
  ;; the one that was expected.
  (nso-t-gt "standardised, the head fits this geometry completely"
            (nso-choice-accuracy m-std stdized) 0.95))

;;; --- shared low-rank projection ------------------------------------------

(let* ((rng (nso-rng 31))
       (options (let (o) (dotimes (_ 4 (nreverse o)) (push (ct--unit rng ct--dim) o))))
       (examples (ct--set rng options 12 0.5))
       (model (nso-choice-lowrank-make ct--dim 3))
       (l2 0.02) (g (nso-choice-lowrank-grad model examples l2))
       (eps 1.0e-5) (worst 0.0))
  ;; Probe every entry in every row: a missing transpose in the shared map
  ;; otherwise still produces a plausible-looking training curve.
  (let ((rows (plist-get model :p)) (grads (plist-get g :dp)) (r 0))
    (dolist (row rows)
      (let ((j 0))
        (dotimes (_ (length row))
          (let ((orig (aref row j)))
            (aset row j (+ orig eps))
            (let ((lp (nso-choice-lowrank-loss model examples l2)))
              (aset row j (- orig eps))
              (let ((lm (nso-choice-lowrank-loss model examples l2)))
                (aset row j orig)
                (setq worst (max worst (/ (abs (- (aref (nth r grads) j)
                                                   (/ (- lp lm) (* 2 eps))))
                                          (max 1.0e-8 (abs (aref (nth r grads) j))))))))
          (setq j (1+ j))))
      (setq r (1+ r))))
  (nso-t-lt "low-rank dL/dP matches finite differences" worst 1.0e-5)))

(let* ((rng (nso-rng 37))
       (opts (let (o) (dotimes (_ 5 (nreverse o)) (push (ct--unit rng ct--dim) o))))
       (train (ct--set rng opts 80 0.8))
       (model (nso-choice-lowrank-train train 4 300 0.5 0.02)))
  (nso-t-gt "low-rank trainer learns the synthetic task"
            (nso-choice-lowrank-accuracy model train) 0.7)
  (nso-t-lt "low-rank trainer reports convergence"
             (plist-get model :final-gnorm) nso-choice-gtol))

(let* ((rng (nso-rng 41)) (dim 1024)
       (opts (let (o) (dotimes (_ 4 (nreverse o)) (push (ct--unit rng dim) o))))
       (examples (let (out)
                   (dotimes (i 24 (nreverse out))
                     (let* ((o (nth (mod i 4) opts)) (s (make-vector dim 0.0)))
                       (dotimes (j dim)
                         (aset s j (* 4.0 (+ (aref o j)
                                             (* 0.1 (- (nso-rng-float rng) 0.5))))))
                       (push (list :state s :options opts :label (mod i 4)) out)))))
       (model (nso-choice-lowrank-train examples 4 250 0.5 0.02)))
  (nso-t-lt "low-rank trainer converges at dimension 1024"
             (plist-get model :final-gnorm) nso-choice-gtol))

;;; choice-test.el ends here
(nso-t-done "choice")
;;; choice-test.el ends here
