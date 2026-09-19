;;; type-gate-test.el --- the gate behind "cannot produce a type error" -*- lexical-binding: t; -*-

;;; Commentary:

;; Section 1 of the design doc claims an ill-typed answer is unrepresentable
;; rather than rare.  This suite is the entire weight behind that claim, so
;; most of it is ill-typed answers that must be rejected.  The happy path is
;; checked first: a gate that rejects everything would pass every negative
;; control and be worthless.

;;; Code:

(require 'nso-types)
(require 'nso-stub)
(load (expand-file-name "nso-test-helper.el"
                        (file-name-directory (or load-file-name buffer-file-name))))

(message "== type gate ==")

(let ((q (nso-make-choice "what is the status of this record?"
                          '(open closed expired))))

  ;; --- the gate must accept a correct answer ------------------------------

  (nso-t-green "a well-formed Choice answer is accepted"
               (nso-type-gate q '(:choice open
                                  :probabilities ((open . 0.7) (closed . 0.2)
                                                  (expired . 0.1))
                                  :confidence 0.7)))

  (nso-t-green "the valid stub answer is accepted"
               (nso-type-gate q (nso-stub-answer-valid q)))

  (nso-t-green "confidence may be omitted"
               (nso-type-gate q '(:choice open
                                  :probabilities ((open . 0.7) (closed . 0.2)
                                                  (expired . 0.1)))))

  ;; --- negative controls: each must come out red --------------------------

  (nso-t-red "an answer outside the declared option set is rejected"
             (nso-type-gate q (nso-stub-answer-out-of-set q)))

  (nso-t-red "an answer outside the set is rejected even when spelled by hand"
             (nso-type-gate q '(:choice pending
                                :probabilities ((open . 0.7) (closed . 0.2)
                                                (expired . 0.1)))))

  (nso-t-red "a distribution that does not sum to 1 is rejected"
             (nso-type-gate q '(:choice open
                                :probabilities ((open . 0.7) (closed . 0.2)
                                                (expired . 0.4)))))

  (nso-t-red "a NaN probability is rejected"
             (nso-type-gate q (list :choice 'open
                                    :probabilities (list (cons 'open (/ 0.0 0.0))
                                                         (cons 'closed 0.2)
                                                         (cons 'expired 0.1)))))

  (nso-t-red "a negative probability is rejected"
             (nso-type-gate q '(:choice open
                                :probabilities ((open . 1.1) (closed . -0.1)
                                                (expired . 0.0)))))

  (nso-t-red "a missing option is rejected"
             (nso-type-gate q '(:choice open
                                :probabilities ((open . 0.8) (closed . 0.2)))))

  (nso-t-red "a probability for an undeclared option is rejected"
             (nso-type-gate q '(:choice open
                                :probabilities ((open . 0.6) (closed . 0.2)
                                                (expired . 0.1) (pending . 0.1)))))

  (nso-t-red "an answer that is not its own distribution's argmax is rejected"
             (nso-type-gate q '(:choice expired
                                :probabilities ((open . 0.7) (closed . 0.2)
                                                (expired . 0.1)))))

  (nso-t-red "a confidence outside [0,1] is rejected"
             (nso-type-gate q '(:choice open
                                :probabilities ((open . 0.7) (closed . 0.2)
                                                (expired . 0.1))
                                :confidence 1.4))))

;; --- ill-formed questions ------------------------------------------------

(nso-t-red "an empty option set is rejected"
           (nso-type-gate (nso-make-choice "?" '())
                          '(:choice open :probabilities ((open . 1.0)))))

(nso-t-red "duplicate options are rejected"
           (nso-type-gate (nso-make-choice "?" '(open open closed))
                          '(:choice open :probabilities ((open . 1.0)))))

(nso-t-red "a cardinality above the cap is rejected"
           (let ((options nil) (i 0))
             (while (< i (1+ nso-max-cardinality))
               (push (intern (format "opt%d" i)) options)
               (setq i (1+ i)))
             (nso-type-gate (nso-make-choice "?" options)
                            (list :choice (car options)
                                  :probabilities (list (cons (car options) 1.0))))))

(nso-t-green "exactly the cap is accepted"
             (let ((options nil) (i 0) (probs nil))
               (while (< i nso-max-cardinality)
                 (push (intern (format "opt%d" i)) options)
                 (setq i (1+ i)))
               (setq options (nreverse options))
               ;; one option at 0.745, the other 254 sharing the rest exactly
               (let ((tail (/ 0.001 1.0)))
                 (push (cons (car options) (- 1.0 (* tail (1- nso-max-cardinality))))
                       probs)
                 (dolist (o (cdr options)) (push (cons o tail) probs)))
               (nso-type-gate (nso-make-choice "?" options)
                              (list :choice (car options)
                                    :probabilities (nreverse probs)))))

;; --- Noul and Score ------------------------------------------------------

(let ((q (nso-make-noul "is this invoice overdue?")))
  (nso-t-green "a well-formed Noul answer is accepted"
               (nso-type-gate q '(:p-yes 0.82 :confidence 0.82)))
  (nso-t-red "a p-yes above 1 is rejected"
             (nso-type-gate q '(:p-yes 1.5)))
  (nso-t-red "a NaN p-yes is rejected"
             (nso-type-gate q (list :p-yes (/ 0.0 0.0))))
  (nso-t-red "a missing p-yes is rejected"
             (nso-type-gate q '(:confidence 0.5))))

(let ((q (nso-make-score "how complete is this record?" '(poor fair good))))
  (nso-t-green "a well-formed Score answer is accepted"
               (nso-type-gate q '(:choice good
                                  :probabilities ((poor . 0.1) (fair . 0.3)
                                                  (good . 0.6)))))
  (nso-t-red "a level outside the legend is rejected"
             (nso-type-gate q '(:choice excellent
                                :probabilities ((poor . 0.1) (fair . 0.3)
                                                (good . 0.6))))))

(nso-t-done "type gate")

;;; type-gate-test.el ends here
