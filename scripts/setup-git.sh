#!/bin/bash
set -e

# Configure username and password
git config user.name "${GITLAB_USER_LOGIN}"
git config user.email "${GITLAB_USER_EMAIL}"

# Use https for git
git config --global url.https://gitlab.com/.insteadOf git@gitlab.com:

# Set up git credentials
git config --global credential.helper store
echo https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.com > ~/.git-credentials

# Set the git remote and checkout the branch
git remote set-url origin https://${GITLAB_USER_LOGIN}:${PROJECT_TOKEN}@gitlab.com/${CI_PROJECT_PATH}
git checkout -b $CI_COMMIT_BRANCH
