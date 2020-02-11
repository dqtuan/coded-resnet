"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

from init import *

if args.deterministic:
    print("MODEL NOT FULLY DETERMINISTIC")
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic=True

# 'cifar10':
trainset = torchvision.datasets.CIFAR10(
    root='../../data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='../../data', train=False, download=True, transform=transform_test)
args.nClasses = 10
in_shape = (3, 32, 32)


if args.deterministic:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                              shuffle=True, num_workers=2, worker_init_fn=np.random.seed(1234))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                             shuffle=False, num_workers=2, worker_init_fn=np.random.seed(1234))
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)

model = iResNet(nBlocks=args.nBlocks,
                nStrides=args.nStrides,
                nChannels=args.nChannels,
                nClasses=args.nClasses,
                init_ds=args.init_ds,
                inj_pad=args.inj_pad,
                in_shape=in_shape,
                coeff=args.coeff,
                numTraceSamples=args.numTraceSamples,
                numSeriesTerms=args.numSeriesTerms,
                n_power_iter = args.powerIterSpectralNorm,
                density_estimation=args.densityEstimation,
                actnorm=(not args.noActnorm),
                learn_prior=(not args.fixedPrior),
                nonlin=args.nonlin)


# init actnrom parameters
init_batch = get_init_batch(trainloader, args.init_batch)
print("initializing actnorm parameters...")
with torch.no_grad():
    model(init_batch, ignore_logdet=True)
print("initialized")

use_cuda = torch.cuda.is_available()
model.cuda()
# TODO:
model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
cudnn.benchmark = True
in_shapes = model.module.get_in_shapes()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# log
# setup logging with visdom
viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
assert viz.check_connection(), "Could not make visdom"

with open(param_path, 'w') as f:
    f.write(json.dumps(args.__dict__))
train_log = open(trainlog_path, 'w')
try_make_dir(args.save_dir)
try_make_dir(sample_path)


# optionally resume from a checkpoint
if args.flag_retrain:
    model_path = os.path.join(checkpoint_path, 'model_e{}.pkt'.format(args.resume))
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_objective = checkpoint['objective']
        print('objective: '+str(best_objective))

        model = checkpoint['model']
        model.module.set_num_terms(args.numSeriesTerms)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

# mode:
if args.analysisTraceEst:
    anaylse_trace_estimation(model, testset, use_cuda, args.extension)
    sys.exit('Done')
if args.norm:
    test_spec_norm(model, in_shapes, args.extension)
    sys.exit('Done')
if args.interpolate:
    interpolate(model, testloader, testset, start_epoch, use_cuda, best_objective, args.dataset)
    sys.exit('Done')
if args.evaluate:
    test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
    if use_cuda:
        model.module.set_num_terms(args.numSeriesTerms)
    else:
        model.set_num_terms(args.numSeriesTerms)
    model = torch.nn.DataParallel(model.module)
    test(best_objective, args, model, start_epoch, testloader, viz, use_cuda, test_log)
    sys.exit('Done')

########################## Training ##################
print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))

elapsed_time = 0
test_objective = -np.inf

def eval_invertibility(model, train_epoch):
    model.eval()
    correct_b = 0
    correct_b_hat = 0
    match_b = 0
    total = 0
    bs = args.batch // 2
    max_iters = 1
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx >= max_iters:
            break
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)

        out, out_bij = model(inputs)
        # divide out_bij into 2 parts
        z_a = out_bij[:bs, ...]
        z_b = out_bij[bs:, ...]
        z_ab = (z_a + z_b) / 2
        x_c = model.module.inverse(z_ab)
        _, z_c = model(x_c)
        z_b_hat = z_c * 2 - z_a

        # invert
        real_b = inputs[bs:, ...]
        x_b = model.module.inverse(z_b)
        x_b_hat = model.module.inverse(z_b_hat)
        x_b_noise = model.module.inverse(z_b * 1.01)

        # save samples
        save_images(sample_path, train_epoch, bs, real_b, 'real')
        save_images(sample_path, train_epoch, bs, x_b, 'inverse')
        save_images(sample_path, train_epoch, bs, x_b_hat, 'inverse_hat')
        save_images(sample_path, train_epoch, bs, x_b_noise, 'inverse_noise')

        # classification scores
        out_b = model.module.classifier(z_b)
        out_b_hat = model.module.classifier(z_b_hat)
        _, y_b = torch.max(out_b.data, 1)
        _, y_b_hat = torch.max(out_b_hat.data, 1)
        targets = targets[bs:, ...]

        correct_b += y_b.eq(targets.data).sum().item()
        correct_b_hat += y_b_hat.eq(targets.data).sum().item()
        match_b += y_b_hat.eq(y_b.data).sum().item()
        total += bs

    print('Correct %: real {:.4f} reconstruct {:.4f} match {:.4f}'.format(100*correct_b/total, 100*correct_b_hat/total, 100*match_b/total))


for epoch in range(1, 1+args.epochs):
    start_time = time.time()
    train(args, model, optimizer, epoch, trainloader, trainset, viz, use_cuda, train_log)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
    # test inverse function
    eval_invertibility(model, epoch)

# store model
print('Testing model')
test_log = open(testlog_path, 'w')
test_objective = test(test_objective, args, model, epoch, testloader, viz, use_cuda, test_log)
print('* Test results : objective = %.2f%%' % (test_objective))
with open(final_path, 'w') as f:
    f.write(str(test_objective))

# store model afer all
net_path = os.path.join(checkpoint_path, 'model_e{}.pkt'.format(args.epochs))
print('Save checkpoint at ', net_path)
state = {
    'model': model,
    'objective': test_objective,
    'epoch': epoch,
}
torch.save(state, net_path)
