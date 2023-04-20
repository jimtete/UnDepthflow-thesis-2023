#!/bin/bash

if [ $# -gt 1 ]; then
    echo "Usage: $0 [COMMIT_MESSAGE]"
    exit 1
fi

MESSAGE=${1:-"pls work"}

git add .
git commit -m "$MESSAGE"
git push
