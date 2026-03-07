#!/usr/bin/env bash
# Cherry-pick commits from SOURCE_BRANCH onto current branch (or TARGET_BRANCH) one by one,
# running tests after each pick. Stops on first test failure or cherry-pick conflict.
set -e
SOURCE_BRANCH="${1:-fix-c23921a}"
TARGET_BRANCH="${2:-main}"
TEST_CMD="${TEST_CMD:-python3 -m pytest tests/ --tb=no -q}"

echo "Source branch: $SOURCE_BRANCH"
echo "Target branch: $TARGET_BRANCH"
echo "Test command:  $TEST_CMD"
echo ""

# Commits on source that are not on target (oldest first)
COMMITS=$(git log "$TARGET_BRANCH..$SOURCE_BRANCH" --oneline --reverse --format="%h %s")
if [ -z "$COMMITS" ]; then
  echo "No commits to cherry-pick ($TARGET_BRANCH is already up to date with $SOURCE_BRANCH)."
  exit 0
fi

# Switch to target if not already
CURRENT=$(git branch --show-current)
if [ "$CURRENT" != "$TARGET_BRANCH" ]; then
  echo "Checking out $TARGET_BRANCH..."
  git checkout "$TARGET_BRANCH"
else
  echo "Already on $TARGET_BRANCH."
fi

COUNT=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  SHA="${line%% *}"
  MSG="${line#* }"
  COUNT=$((COUNT + 1))
  echo "----------------------------------------"
  echo "[$COUNT] Cherry-picking: $SHA $MSG"
  echo "----------------------------------------"
  if ! git cherry-pick "$SHA"; then
    echo "Cherry-pick failed (conflict?). Resolve and run tests, or: git cherry-pick --abort"
    exit 1
  fi
  echo "Running tests..."
  if ! $TEST_CMD; then
    echo "Tests failed after cherry-pick $SHA. To undo: git reset --hard HEAD~1"
    exit 1
  fi
  echo "OK"
  echo ""
done <<EOF
$COMMITS
EOF

echo "Done: $COUNT commit(s) cherry-picked and verified."
