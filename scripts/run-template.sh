#!/bin/bash

set -e

# Set the repository path of the current GitLab project
export repo_path=$(echo $CI_PROJECT_PATH | sed -e 's/namespace\///')


# Run cruft with the provided parameters
export CRUFT_ARGS="{ 
    \"name\": \"$name\", 
    \"description\": \"$description\", 
    \"repo_path\": \"$repo_path\", 
    \"author_name\": \"$author_name\", 
    \"author_email\": \"$author_email\", 
    \"use_renovate\": \"$use_renovate\"
}"


cruft create \
    --no-input \
    --extra-context "$CRUFT_ARGS" \
    git@gitlab.com:namespace/group/backend/pypackage-skeleton.git


# Clean-up stuff that is not part of the generated project, and move
# generated project into the root of the repository
export slug=$(echo $CI_PROJECT_NAME | sed -e 's/[-\ ]/\_/g')
shopt -s dotglob
rm -rf scripts
mv ${slug}/* .
rm -rf $slug media

# Make our first proper commit with the new project
git add .
git commit -m "chore: initial commit"
git push origin main
