;;; compile-deps.el --- byte-compile the sibling lisp we run, into our own tree -*- lexical-binding: t; -*-

;; The encode stage spends most of its time in Elisp that belongs to the
;; sibling repositories -- RMSNorm, the rotation, attention, SwiGLU -- and
;; those repositories ship no .elc, so `-L ../nelisp-llm/lisp' loads and
;; interprets the source.  Interpreted, the numeric loops here run at about
;; 3.3M float operations a second; byte-compiled they are several times that.
;;
;; The output goes into build/elc rather than next to the sources.  Writing
;; .elc into another repository's tree leaves artefacts that outlive this work
;; and can shadow an edited .el there later, which is a failure mode with
;; history in this project.  Redirecting keeps their trees clean and makes the
;; cache disposable: `make clean' removes it and nothing is stale afterwards.
;;
;; Run:  emacs -Q --batch -l tools/compile-deps.el

(defvar nso-cd--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(defvar nso-cd--out (expand-file-name "../build/elc" nso-cd--here))

(unless (file-directory-p nso-cd--out) (make-directory nso-cd--out t))

(dolist (d '("nelisp-photon/lisp" "nelisp-gpu/lisp" "nelisp-llm/lisp"))
  (add-to-list 'load-path (expand-file-name (concat "../../" d) nso-cd--here)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-cd--here))
(add-to-list 'load-path nso-cd--out)

(setq byte-compile-dest-file-function
      (lambda (src)
        (expand-file-name (concat (file-name-base src) ".elc") nso-cd--out)))

;; Warnings from another project's sources are not this project's business and
;; would bury the one line that matters.
(setq byte-compile-warnings nil
      byte-compile-verbose nil)

(let ((n 0) (failed nil))
  ;; nelisp-gpu is deliberately absent.  It locates the vkserver binary
  ;; relative to its own source file, and a redirected .elc makes
  ;; `load-file-name' point at build/elc, so the lookup fails with "Doing
  ;; vfork: No such file or directory".  It is also the one layer where
  ;; compiling buys nothing: its time goes into pipe I/O, not arithmetic.
  (dolist (dir (list (expand-file-name "../../nelisp-photon/lisp" nso-cd--here)
                     (expand-file-name "../../nelisp-llm/lisp" nso-cd--here)))
    (when (file-directory-p dir)
      (dolist (f (directory-files dir t "\\.el\\'"))
        (condition-case err
            (when (byte-compile-file f) (setq n (1+ n)))
          (error (push (cons (file-name-nondirectory f) err) failed))))))
  (princ (format "compiled %d sibling files into %s\n" n nso-cd--out))
  (when failed
    ;; Not fatal: a file that will not compile is simply loaded from source, as
    ;; it was before.  Saying so beats a silent partial cache.
    (princ (format "%d did not compile and will load from source:\n" (length failed)))
    (dolist (f failed) (princ (format "  %s\n" (car f))))))

;;; compile-deps.el ends here
