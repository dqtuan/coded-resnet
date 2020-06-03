# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/src/vae/arts/ --exclude=data ./results/

<<<<<<< HEAD
rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/pix2pix ./results/
=======
# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/pix2pix ./results/ --exclude=checkpoints

<<<<<<< HEAD
rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/fusion ./results/ --exclude=checkpoints

# rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results/iresnet ./results/ --exclude=checkpoints
>>>>>>> 813c6a91d35527ab115f0db80461d4a811bb3f73
=======
rsync -av -e 'ssh -A tuandinh@144.92.237.175 ssh' tuandinh@128.104.158.78:~/Documents/Projects/1912-invnet/results ./ --exclude=checkpoints
>>>>>>> de56206e152ffdcd4259d319dd41f4f8441138cd
