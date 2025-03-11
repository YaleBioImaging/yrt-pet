Git feature development workflow
================================

-  The main branch is the stable branch of the project, it is designed
   to remain “clean” (i.e. tested and documented).
-  Work on the project should be performed on *branches*. A branch is
   created for each feature, bug, etc. and progress is committed on that
   branch.
-  Once the work is ready to *merge* (i.e. the code is complete, tested
   and documented) it can be submitted for review: the process is called
   *merge request*. Follow Github’s instructions to submit a pull
   request.
-  The request is then reviewed by one of the project’s maintainers.
   After discussions, it should eventually be merged to the main branch
   by the maintainer.
-  The branch on which the work was performed can then be deleted (the
   history is preserved). The branch commits will be squashed before the
   merge (this can be done in Github’s web interface).
-  If the main branch has changed between the beginning of the work and
   the pull request submission, the branch should be *rebased* to the
   main branch (in practice, tests should be run after a rebase to avoid
   potential regressions).

   -  In some cases, unexpected rebase are reported (for instance if the
      history is A–B–C and B is merged to A, a later rebase of C to A
      may cause conflicts that should not exist). In such cases, two
      fixes are possible:

      -  Launching an interactive rebase
         (``git rebase -i <main branch>``) and dropping the commits that
         would be duplicated.

   -  After a rebase, ``git push`` by default will not allow an update
      that is not ``fast-forward`` with the corresponding remote branch,
      causing an error when trying to push.
      ``git push --force-with-lease`` can be used to force a push while
      checking that the remote branch has not changed. Note that this
      will lose history
