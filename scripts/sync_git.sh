sshpass -p $1 git pull origin138 master
git pull origin master
sshpass -p $1 git push origin138 master
git push origin master
