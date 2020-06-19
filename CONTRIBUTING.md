[![Build Status](https://travis-ci.org/hippylib/hippylib.svg?branch=master)](https://travis-ci.org/hippylib/hippylib)
[![Doc Status](https://readthedocs.org/projects/hippylib/badge/?version=latest&style=flat)](https://hippylib.readthedocs.io/en/latest/)
[![status](http://joss.theoj.org/papers/053e0d08a5e9755e7b78898cff6f6208/status.svg)](http://joss.theoj.org/papers/053e0d08a5e9755e7b78898cff6f6208) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.596931.svg)](https://doi.org/10.5281/zenodo.596931)

                  Inverse Problem PYthon library

```
 __        ______  _______   _______   __      __  __  __  __       
/  |      /      |/       \ /       \ /  \    /  |/  |/  |/  |      
$$ |____  $$$$$$/ $$$$$$$  |$$$$$$$  |$$  \  /$$/ $$ |$$/ $$ |____  
$$      \   $$ |  $$ |__$$ |$$ |__$$ | $$  \/$$/  $$ |/  |$$      \ 
$$$$$$$  |  $$ |  $$    $$/ $$    $$/   $$  $$/   $$ |$$ |$$$$$$$  |
$$ |  $$ |  $$ |  $$$$$$$/  $$$$$$$/     $$$$/    $$ |$$ |$$ |  $$ |
$$ |  $$ | _$$ |_ $$ |      $$ |          $$ |    $$ |$$ |$$ |__$$ |
$$ |  $$ |/ $$   |$$ |      $$ |          $$ |    $$ |$$ |$$    $$/ 
$$/   $$/ $$$$$$/ $$/       $$/           $$/     $$/ $$/ $$$$$$$/  
```                                                                    
                                                                    

                  https://hippylib.github.io
                  
# How to Contribute

The `hIPPYlib` team welcomes contributions at all levels: bugfixes, code
improvements, new capabilities, improved documentation, 
or new examples/tutorials.

Use a pull request (PR) toward the `hippylib:master` branch to propose your
contribution. If you are planning significant code changes, or have any
questions, you should also open an [issue](https://github.com/hippylib/hippylib/issues)
before issuing a PR. 

See the [Quick Summary](#quick-summary) section for the main highlights of our
GitHub workflow. For more details, consult the following sections and refer
back to them before issuing pull requests:

- [GitHub Workflow](#github-workflow)
  - [hIPPYlib Organization](#hippylib-organization)
  - [New Feature Development](#new-feature-development)
  - [Developer Guidelines](#developer-guidelines)
  - [Pull Requests](#pull-requests)
  - [Pull Request Checklist](#pull-request-checklist)
- [Automated Testing](#automated-testing)
- [Contact Information](#contact-information)

Contributing to hIPPYlib requires knowledge of Git and, likely, inverse problems.
If you are new to Git, see the [GitHub learning
resources](https://help.github.com/articles/git-and-github-learning-resources/).
To learn more about inverse problems, see our [tutorial page](http://hippylib.github.io/tutorial).

*By submitting a pull request, you are affirming the* [Developer's Certificate of
Origin](#developers-certificate-of-origin-11) *at the end of this file.*


## Quick Summary

- We encourage you to [join the hIPPYlib organization](#hippylib-organization) and create
  development branches off `hippylib:master`.
- Please follow the [developer guidelines](#developer-guidelines), in particular
  with regards to documentation and code styling.
- Pull requests  should be issued toward `hippylib:master`. Make sure
  to check the items off the [Pull Request Checklist](#pull-request-checklist).
- After approval, hIPPYlib developers merge the PR in `hippylib:master`.
- Don't hesitate to [contact us](#contact-information) if you have any questions.


## GitHub Workflow

The GitHub organization, https://github.com/hippylib, is the main developer hub for
the hIPPYlib project.

If you plan to make contributions or will like to stay up-to-date with changes
in the code, *we strongly encourage you to [join the hIPPYlib organization](#hippylib-organization)*.

This will simplify the workflow (by providing you additional permissions), and
will allow us to reach you directly with project announcements.


### hIPPYlib Organization

- Before you can start, you need a GitHub account, here are a few suggestions:
  + Create the account at: github.com/join.
  + For easy identification, please add your name and maybe a picture of you at: https://github.com/settings/profile.
  + To receive notification, set a primary email at: https://github.com/settings/emails.
  + For password-less pull/push over SSH, add your SSH keys at: https://github.com/settings/keys.

- [Contact us](#contact-information) for an invitation to join the hIPPYlib GitHub
  organization.

- You should receive an invitation email, which you can directly accept.
  Alternatively, *after logging into GitHub*, you can accept the invitation at
  the top of https://github.com/hippylib.

- Consider making your membership public by going to https://github.com/orgs/hippylib/people
  and clicking on the organization visibility dropbox next to your name.

- Project discussions and announcements will be posted at
  https://github.com/orgs/hippylib/teams/everyone.

- The hIPPYlib source code is in the [hippylib](https://github.com/hippylib/hippylib)
  repository.

- The website is in the [web](https://github.com/hippylib/web) repository.



### New Feature Development

- A new feature should be important enough that at least one person, the
  proposer, is willing to work on it and be its champion.

- The proposer creates a branch for the new feature (with suffix `-dev`), off
  the `master` branch, or another existing feature branch, for example:

  ```
  # Clone assuming you have setup your ssh keys on GitHub:
  git clone git@github.com:hippylib/hippylib.git

  # Alternatively, clone using the "https" protocol:
  git clone https://github.com/hippylib/hippylib.git

  # Create a new feature branch starting from "master":
  git checkout master
  git pull
  git checkout -b feature-dev

  # Work on "feature-dev", add local commits
  # ...

  # (One time only) push the branch to github and setup your local
  # branch to track the github branch (for "git pull"):
  git push -u origin feature-dev

  ```

- **We prefer that you create the new feature branch as a fork.**
  To allow hIPPYlib developers to edit the PR, please [enable upstream edits](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/).

- The typical feature branch name is `new-feature-dev`, e.g. `optimal_exp_design-dev`. While
  not frequent in hIPPYlib, other suffixes are possible, e.g. `-fix`, `-doc`, etc.


### Developer Guidelines

- *Keep the code lean and as simple as possible*
  - Well-designed simple code is frequently more general and powerful.
  - Lean code base is easier to understand by new collaborators.
  - New features should be added only if they are necessary or generally useful.
  - Code must be compatible with Python 3.
  - When adding new features add an example in the `application` folder and/or a
    new notebook in the `tutorial` folder.
  - The preferred way to export solutions for visualization in paraview is using `dl.XDMFFile`

- *Keep the code general and reasonably efficient*
  - Main goal is fast prototyping for research.
  - When in doubt, generality wins over efficiency.
  - Respect the needs of different users (current and/or future).

- *Keep things separate and logically organized*
  - General usage features go in hIPPYlib (implemented in as much generality as
    possible), non-general features go into external apps/projects.
  - Inside hIPPYlib, compartmentalize between modeling, algorithms, utils, etc.
  - Contributions that are project-specific or have external dependencies are
    allowed (if they are of broader interest), but should be `#ifdef`-ed and not
    change the code by default.

- Code specifics
  - All significant new classes, methods and functions have sphinx-style
    documentation in source comments.
  - Code styling should resemble existing code.
  - When manually resolving conflicts during a merge, make sure to mention the
    conflicted files in the commit message.

### Pull Requests

- When your branch is ready for other developers to review / comment on
  the code, create a pull request towards `hippylib:master`.

- Pull request typically have titles like:

     `Description [new-feature-dev]`

  for example:

     `Bayesian Optimal Design of Experiments [oed-dev]`

  Note the branch name suffix (in square brackets).

- Titles may contain a prefix in square brackets to emphasize the type of PR.
  Common choices are: `[DON'T MERGE]`, `[WIP]` and `[DISCUSS]`, for example:

     `[DISCUSS] Bayesian Optimal Design of Experiments [oed-dev]`

- Add a description, appropriate labels and assign yourself to the PR. The hIPPYlib
  team will add reviewers as appropriate.

- List outstanding TODO items in the description.

- Track the Travis CI [continuous integration](#automated-testing)
  builds at the end of the PR. These should run clean, so address any errors as
  soon as possible.


### Pull Request Checklist

Before a PR can be merged, it should satisfy the following:

- [ ] CI runs without errors.
- [ ] Update `CHANGELOG`:
    - [ ] Is this a new feature users need to be aware of? New or updated application or tutorial?
    - [ ] Does it make sense to create a new section in the `CHANGELOG` to group with other related features?
- [ ] New examples/applications/tutorials:
    - [ ] All new examples/applications/tutorials run as expected.
    - [ ] Add a *fast version* of the example/application/tutorial to Travis CI
- [ ] New capability:
   - [ ] All significant new classes, methods and functions have sphinx-style documentation in source comments.
   - [ ] Add new examples/applications/tutorials to highlight the new capability.
   - [ ] For new classes, functions, or modules, edit the corresponding `.rst` file in the `doc` folder.
   - [ ] If this is a major new feature, consider mentioning in the short summary inside `README` *(rare)*.
   - [ ] If this is a `C++` extension, the `package_data` dictionary in `setup.py` should include new files.


## Automated Testing

We use Travis CI to drive the default tests on the `master` and `feature`
branches. See the `.travis` file and the logs at
[https://travis-ci.org/hi/hippylib](https://travis-ci.org/hippylib/hippylib).

Testing using Travis CI should be kept lightweight, as there is a 50 minute time
constraint on jobs.

- Tests on the `master` branch are triggered whenever a push is issued on this branch.


## Contact Information

- Contact the hIPPYlib team by posting to the [GitHub issue tracker](https://github.com/hippylib/hippylib/issues).
  Please perform a search to make sure your question has not been answered already.
  
## Slack channel

The hIPPYlib slack channel is a good resource to request and receive help with using hIPPYlib. Everyone is invited to read and take part in discussions. Discussions about development of new features in hIPPYlib also take place here. You can join our Slack community by filling in [this form](https://forms.gle/w8B7uKSXxdVCmfZ99). 

## [Developer's Certificate of Origin 1.1](https://developercertificate.org/)

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right
    to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my
    knowledge, is covered under an appropriate open source license and I have
    the right under that license to submit that work with modifications, whether
    created in whole or in part by me, under the same open source license
    (unless I am permitted to submit under a different license), as indicated in
    the file; or

(c) The contribution was provided directly to me by some other person who
    certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I
    submit with it, including my sign-off) is maintained indefinitely and may be
    redistributed consistent with this project or the open source license(s)
    involved.
    
---    
> *Acknowledgement*: We thank the [MFEM team](https://github.com/mfem) for allowing us to use their
contributing guidelines file as template.
