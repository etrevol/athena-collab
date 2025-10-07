#!/bin/bash
set -e  # stop the script immediately if any command fails

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "You must provide a commit message as an argument."
  exit 1
fi

# Check if there are any changes to commit
if git diff-index --quiet HEAD --; then
  echo "No changes to commit."
  exit 0
fi

# Add all changes to the staging area
echo "Adding all changes to the staging area..."
git add .

# Create a commit with the provided message
echo "Creating commit with message: $*"
git commit -m "$*"

# Push changes to the remote repository
echo "Pushing changes to the remote repository..."
git push origin $(git branch --show-current)

echo "Commit and push completed successfully!"