import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
from torchvision import transforms
import torch
def eval(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False,
         alpha=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]

            input = input.cuda()
            output = model(input)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output = torch.clamp(output.cuda(), 0, 1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]

        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)