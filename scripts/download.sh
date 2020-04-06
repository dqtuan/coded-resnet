# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/src/vae/arts/ --exclude=data ./results/

# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/pix2pix ./results/ --exclude=checkpoints

rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/fusion ./results/ --exclude=checkpoints

rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/iresnet ./results/ --exclude=checkpoints
