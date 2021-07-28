import torch
import lpips

def get_gan_losses_fn():
    bce = torch.nn.BCEWithLogitsLoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(r_logit, torch.ones_like(r_logit))
        f_loss = bce(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn

def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = torch.max(1 - f_logit, torch.zeros_like(f_logit)).mean()
        return f_loss

    return d_loss_fn, g_loss_fn

def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1- r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1+ f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss
    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss
    return d_loss_fn, g_loss_fn

def get_lsgan_losses_fn(): #这个写的有点问题，应该不是ones和zeros
    mse = torch.nn.MSELoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(r_logit, torch.ones_like(r_logit))
        f_loss = mse(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn

def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = -r_logit.mean()
        f_loss = f_logit.mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()


def multiScale_loss(x,x_):
    loss_mse = torch.nn.MSELoss()
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')
    loss_kl = torch.nn.KLDivLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    l1 = loss_mse(x,x_)

    # logit_x, logit_x_ = torch.nn.functional.softmax(x), torch.nn.functional.softmax(x_)
    # l2 = loss_kl(torch.log(x_),x) # True：x, Flase: x_.
    # l2 = torch.where(torch.isnan(l2),torch.full_like(l2,0),l2)
    # l2 = torch.where(torch.isinf(l2),torch.full_like(l2,1),l2)

    #vector_x, vector_x_ = x.view(-1), x_.view(-1)
    #l2 = abs(1-vector_x.dot(vector_x_)/(torch.sqrt(vector_x.dot(vector_x))*torch.sqrt(vector_x_.dot(vector_x_))))
    l2 = (1-abs(torch.cosine_similarity(x.view(x.shape[0],-1),x_.view(x_.shape[0],-1)))).mean()

    l3 = loss_lpips(x,x_).mean()

    #l4 = loss_ce(x.long(),x_.long())

    print('l1,l2,l3,l4,l5:')
    print(l1)
    print(l2)
    print(l3)
    #print(l4)

    l = l1+l2+l3
    return l

