#!/usr/bin/env bash
# ==============================================================================
# push_to_github.sh
# Automates private GitHub repository creation, remote configuration, and pushing
# for Assignment 1B: Enterprise Domain-Specific LLM.
# ==============================================================================

set -e

REPO_NAME="enterprise-domain-llm"
REPO_DESC="End-to-end enterprise domain-specific LLM adaptation, QLoRA fine-tuning, inference optimization, speculative decoding, 4-bit quantization and production benchmarking for Assignment 1B."

echo "======================================================================"
echo "GITHUB PRIVATE REPOSITORY AUTOMATION — ENTERPRISE DOMAIN LLM"
echo "======================================================================"

# 1. Verify gh CLI authentication
if ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI is not authenticated."
    echo "Please run: gh auth login"
    echo "Follow the prompts (GitHub.com -> HTTPS -> Web Browser), then re-run this script."
    exit 1
fi

GH_USER=$(gh api user -q .login)
echo "Authenticated as GitHub user: @${GH_USER}"

# 2. Check if repository already exists
EXISTS=$(gh repo view "${GH_USER}/${REPO_NAME}" --json name -q .name 2>/dev/null || echo "")

if [ -z "$EXISTS" ]; then
    echo "Repository '${REPO_NAME}' does not exist on GitHub. Creating PRIVATE repository..."
    gh repo create "${REPO_NAME}" --private --description "${REPO_DESC}"
    echo "Successfully created private repository: https://github.com/${GH_USER}/${REPO_NAME}"
else
    echo "Repository '${REPO_NAME}' already exists on GitHub."
    IS_PRIVATE=$(gh repo view "${GH_USER}/${REPO_NAME}" --json isPrivate -q .isPrivate)
    if [ "$IS_PRIVATE" != "true" ]; then
        echo "SECURITY WARNING: Repository is currently PUBLIC! Making it PRIVATE..."
        gh repo edit "${GH_USER}/${REPO_NAME}" --visibility private
    else
        echo "Verified: Repository is already PRIVATE."
    fi
fi

# 3. Configure git remote
cd "$(dirname "$0")"

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "https://github.com/${GH_USER}/${REPO_NAME}.git"
else
    git remote add origin "https://github.com/${GH_USER}/${REPO_NAME}.git"
fi

# 4. Push branch
git branch -M main
echo "Pushing commits to private repository..."
git push -u origin main

# 5. Final verification
echo ""
echo "======================================================================"
echo "VERIFICATION REPORT"
echo "======================================================================"
gh repo view "${GH_USER}/${REPO_NAME}" --json name,isPrivate,url,visibility
echo "Latest remote commit:"
git log -1 --oneline
echo "SUCCESS: Project is live and securely private on GitHub!"
