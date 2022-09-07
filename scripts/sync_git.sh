



gitdiff=$(git diff --name-only)
if [ "$gitdiff" ]; then
    git add -u .
    IFS="*^&" # Change the IFS temporarily
    read -p "Found code changes, git added, now git commit -- please enter the git commit message: " gcm
    IFS=" "
    git commit -m "$gcm"
fi

tocommit=$(git status | grep "Changes to be committed:")
if [ "$tocommit" ]; then
    IFS="*^&" # Change the IFS temporarily
    read -p "Found uncommitted changes, please enter the git commit message: " gcm
    IFS=" "
    git commit -m "$gcm"
fi
echo "==> Check code change done!"

current_branch=$(git rev-parse --abbrev-ref HEAD)

echo "==> git pull origin $current_branch"
git pull origin $current_branch

echo "==> git pull origin138 $current_branch"
read -p "Please input passwd for origin138: " passwd
sshpass -p $passwd git pull origin138 $current_branch

echo "==> git push origin $current_branch"
git push origin $current_branch

echo "==> git push origin138 $current_branch"
sshpass -p $passwd git push origin138 $current_branch
