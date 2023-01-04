# Contributing

Welcome to the PyTorch Lightning community! We're building the most advanced research platform on the planet to implement the latest, best practices that the amazing PyTorch team rolls out!

## Design Principles

We encourage all sorts of contributions you're interested in adding! When writing Tutorials, please follow these principles.

#### Simple Internal Code

It's useful for users to look at the code and understand very quickly what's happening.
Many users won't be engineers. Thus we need to value clear, simple code over condensed ninja moves.
While that's super cool, this isn't the project for that :)

#### Force User Decisions To Best Practices

There are 1,000 ways to do something. However, eventually one popular solution becomes standard practice, and everyone follows.
We try to find the best way to solve a particular problem, and then force our users to use it for readability and simplicity.

When something becomes a best practice, we add it to the framework. This is usually something like bits of code in utils or in the model file that everyone keeps adding over and over again across projects. When this happens, bring that code inside the trainer and add a flag for it.

#### Gain User Trust

As a researcher, you can't have any part of your code going wrong. So, make thorough tests to ensure that every implementation of a new trick or subtle change is correct.

#### Interoperability

PyTorch Lightning Tutorials is highly interoperable with PyTorch Lightning and PyTorch.

______________________________________________________________________

## Contribution Types

We are always looking for help to implement new features or fixing bugs.

A lot of good work has already been done in project mechanics (\_requirements/base.txt, setup.py, pep8, badges, ci, etc...) so we're in a good state there thanks to all sooner contributors!

### Bug Fixes:

1. If you find a bug please submit a GitHub issue. Make sure the title explains the issue.
1. Try to fix it or recommend a solution.
1. Submit a PR!

_**Note**, even if you do not find the solution, sending a PR with a test covering the issue is a valid contribution, and we can help you or finish it with you :\]_

### New Models:

PyTorch Lightning Tutorials shows several research models for ready usage. Following are general guidelines for adding new models.

1. Workflows which are standard baselines
1. Whose results are reproduced properly either by us or by authors.
1. Do not reinvent the wheel, natively support torchvision, torchtext, torchaudio models.
1. Use open source licensed models.

Please raise an issue before adding a new tutorial. There are tons of models that keep coming. It is very difficult to support every peace.

______________________________________________________________________

## Guidelines

For this section, we refer to read the [parent PL guidelines](https://pytorch-lightning.readthedocs.io/en/stable/CONTRIBUTING.html)

**Reminder**

All added or edited code shall be the own original work of the particular contributor.
If you use some third-party implementation, all such blocks/functions/modules shall be properly referred and if possible also agreed by code's author. For example - `This code is inspired from http://...`.
In case you adding new dependencies, make sure that they are compatible with the actual PyTorch Lightning license (each particular tutorial can have own licence).

### Question & Answer

1. **How can I help/contribute?**

   All help is extremely welcome - reporting bugs, fixing documentation, etc. To solve some issues you can start with label [good first issue](https://github.com/Lightning-AI/lightning-bolts/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or chose something close to your domain with label. Before you start to implement anything check that the issue description that it is clear and self-assign the task to you (if it is not possible, just comment that you take it, and we assign it to you...).

1. **Is there a recommendation for branch names?**

   We do not rely on the name convention so far you are working with your own fork. Anyway it would be nice to follow this convention `<type>/<issue-id>_<short-name>` where the types are: `bugfix`, `ipynb`, `docs`, ...

1. **I have a model in other framework than PyTorch, how do I add it here?**

   Since PL Tutorials are aiming at Pytorch Lightning implementations we encourage staying with it.
