# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' ./results/inverse/checkpoint-old.t7 tuandinh@128.104.158.78:~/Documents/Projects/1912-inet/results/inverse/
rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' ./src tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/ --exclude=results
