;;; nso-stack-paths.el --- find this repo's lisp and its sibling substrate  -*- lexical-binding: t; -*-

;; Five files under tools/ and test/ each defined their own pair of path
;; helpers and then walked the same list of sibling repositories.  This
;; resolves them once, from the location of this file, so the answer cannot
;; drift between a probe and the test that is supposed to check it.
;;
;; Resolution order per dependency, first existing wins:
;;
;;   1. its environment variable.  NSO_LLM_ROOT keeps its existing meaning --
;;      a checkout root whose `lisp' is used -- because the byte-code cache
;;      has to be built from the same tree that will be loaded; that is the
;;      pin the Makefile documents.  NELISP_LLM_LISP / NELISP_PHOTON_LISP /
;;      NELISP_GPU_LISP name a `lisp' directory directly, matching the
;;      sibling repositories;
;;   2. `vendor/<repo>/lisp' inside this repository, where a submodule or a
;;      vendored copy would land;
;;   3. the sibling checkout `../<repo>/lisp', which is the working layout.
;;
;; No `locate-dominating-file' and no `seq': this file is also read by the
;; NeLisp standalone reader, which does not provide the former.

;;; Code:

(defun nso-stack-paths--root-from (dir)
  "Walk up from DIR to the checkout that holds a `lisp' directory."
  (let ((cur (directory-file-name (expand-file-name dir)))
        (found nil)
        (climbing t))
    (while (and climbing (not found))
      (if (file-directory-p (expand-file-name "lisp" cur))
          (setq found cur)
        (let ((up (directory-file-name (file-name-directory cur))))
          (if (equal up cur)
              (setq climbing nil)
            (setq cur up)))))
    (or found (directory-file-name (expand-file-name ".." dir)))))

(defconst nso-stack-paths-root
  (nso-stack-paths--root-from
   (file-name-directory (or load-file-name buffer-file-name default-directory)))
  "Absolute path of the nelisp-system-one checkout this file belongs to.")

(defconst nso-stack-paths-dependencies
  '(("nelisp-llm"    "NELISP_LLM_LISP"    "NSO_LLM_ROOT")
    ("nelisp-photon" "NELISP_PHOTON_LISP")
    ("nelisp-gpu"    "NELISP_GPU_LISP"))
  "Sibling repositories whose `lisp' directory this repo loads from.
Each entry is (REPO-NAME LISP-DIR-VARIABLE [CHECKOUT-ROOT-VARIABLE]).")

(defun nso-stack-paths--first-existing (dirs)
  "Return the first directory in DIRS that exists, or nil."
  (let (found)
    (dolist (dir dirs found)
      (when (and (not found) dir (file-directory-p dir))
        (setq found dir)))))

(defun nso-stack-paths--from-env (var &optional suffix)
  "Expand VAR's value, appending SUFFIX when it names a checkout root."
  (let ((v (and var (getenv var))))
    (when (and v (not (string-empty-p v)))
      (if suffix (expand-file-name suffix (expand-file-name v))
        (expand-file-name v)))))

(defun nso-stack-paths-locate (repo)
  "Return REPO's `lisp' directory, or nil when no candidate exists."
  (let ((entry (assoc repo nso-stack-paths-dependencies)))
    (nso-stack-paths--first-existing
     (list (nso-stack-paths--from-env (nth 1 entry))
           (nso-stack-paths--from-env (nth 2 entry) "lisp")
           (expand-file-name (concat "vendor/" repo "/lisp") nso-stack-paths-root)
           (expand-file-name (concat "../" repo "/lisp") nso-stack-paths-root)))))

;;;###autoload
(defun nso-stack-paths-ensure ()
  "Put this repo's `lisp' and every dependency found on `load-path'.
Returns the directories added or already present.  A missing dependency
is skipped silently -- nelisp-gpu is genuinely absent on a machine
without the backend, and the caller's own `require' is what names one
that should have been there."
  (let ((dirs (list (expand-file-name "lisp" nso-stack-paths-root))))
    (dolist (dep nso-stack-paths-dependencies)
      (let ((dir (nso-stack-paths-locate (car dep))))
        (when dir (push dir dirs))))
    (setq dirs (nreverse dirs))
    (dolist (dir dirs) (add-to-list 'load-path dir))
    dirs))

(nso-stack-paths-ensure)

(provide 'nso-stack-paths)
;;; nso-stack-paths.el ends here
