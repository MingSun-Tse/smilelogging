
git_change=$(git diff --name-only)
if [ $git_change ]; then
    git add -u .
    read -p "git commit -m " gcm
    echo $gcm
    git commit -m $gcm
fi
sshpass -p $1 git pull origin138 master
git pull origin master
sshpass -p $1 git push origin138 master
git push origin master