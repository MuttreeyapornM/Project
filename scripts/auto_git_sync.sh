#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_DIR="$REPO_DIR/.git-auto"
LOG_FILE="$REPO_DIR/auto-git-sync.log"
REMOTE_URL="https://github.com/MuttreeyapornM/Project.git"
BRANCH="main"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG_FILE"
}

exec 9>"$REPO_DIR/.auto-git-sync.lock"
if ! flock -n 9; then
  log "another sync is already running"
  exit 0
fi

git_auto() {
  git --git-dir="$GIT_DIR" --work-tree="$REPO_DIR" "$@"
}

if [ ! -d "$GIT_DIR" ]; then
  git --git-dir="$GIT_DIR" --work-tree="$REPO_DIR" init
  git_auto config user.name "Project Auto Sync"
  git_auto config user.email "project-auto-sync@users.noreply.github.com"
  git_auto remote add origin "$REMOTE_URL"
  git_auto checkout -B "$BRANCH"
  log "initialized auto-sync repository"
fi

if ! git_auto config --get remote.origin.url >/dev/null; then
  git_auto remote add origin "$REMOTE_URL"
elif [ "$(git_auto config --get remote.origin.url)" != "$REMOTE_URL" ]; then
  git_auto remote set-url origin "$REMOTE_URL"
fi

current_branch="$(git_auto symbolic-ref --short HEAD 2>/dev/null || true)"
if [ "$current_branch" != "$BRANCH" ]; then
  git_auto checkout -B "$BRANCH"
fi

remote_branch_exists=0
if git_auto ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  remote_branch_exists=1
  git_auto fetch origin "$BRANCH"
fi

git_auto add -A

if git_auto diff --cached --quiet; then
  log "no file changes"
else
  commit_message="Auto update: $(date '+%Y-%m-%d %H:%M:%S %z')"
  git_auto commit -m "$commit_message"
  log "created commit: $commit_message"
fi

if [ "$(git_auto rev-list --count HEAD 2>/dev/null || echo 0)" -eq 0 ]; then
  log "no commits to push"
  exit 0
fi

if [ "$remote_branch_exists" -eq 1 ] && git_auto rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  remote_ahead="$(git_auto rev-list --count "HEAD..origin/$BRANCH")"
  local_ahead="$(git_auto rev-list --count "origin/$BRANCH..HEAD")"

  if [ "$remote_ahead" -gt 0 ]; then
    if ! git_auto pull --rebase origin "$BRANCH"; then
      log "pull/rebase failed; resolve conflicts manually"
      exit 1
    fi
    log "pulled updates from origin/$BRANCH"
    git_auto fetch origin "$BRANCH"
    remote_ahead="$(git_auto rev-list --count "HEAD..origin/$BRANCH")"
    local_ahead="$(git_auto rev-list --count "origin/$BRANCH..HEAD")"
  fi

  if [ "$local_ahead" -eq 0 ] && [ "$remote_ahead" -eq 0 ]; then
    log "no commits to push"
    exit 0
  fi
fi

if ! git_auto push -u origin "$BRANCH"; then
  log "push failed; configure non-interactive GitHub credentials for systemd"
  exit 1
fi
log "pushed to origin/$BRANCH"
