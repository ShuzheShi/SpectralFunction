import numpy as np
import matplotlib.pyplot as plt
from paras import args


def plotfigure(taui,omegai,truth,out,output,target):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)


    # rho_t =  torch.from_numpy(truth).float().to(device)
    # output = D(taui,omegai,rho_t)

    npomega = omegai.cpu().detach().numpy()
    ax1.plot(npomega,out.cpu().detach().numpy(),'r-',label='reconstruction')
    ax1.plot(npomega,truth,'-.',label='ground truth')
    # ax1.set_xlim([0,10])
    # ax1.set_ylim([0,1])


    ax1.set_title('Reconstructed spectrum') 
    ax1.set_xlabel('$\omega$') 
    ax1.set_ylabel('$\\rho(\omega)$') 
    ax1.grid(axis='both', alpha=.3)
    ax1.set_xlim([0,10])
    ax1.legend()

    ax2.plot(taui.cpu().detach().numpy(),output.cpu().detach().numpy(),'.',label='prediction')
    ax2.plot(taui.cpu().detach().numpy(),target.cpu().detach().numpy(),'r.',label='ground truth')
    ax2.set_title('Predicted correlators')
    ax2.set_xlabel('$\\tau$')  
    ax2.set_ylabel('$D(\\tau)$') 
    ax2.grid(axis='both', alpha=.3)
    ax2.legend()

    ax3.plot(omegai.cpu().detach().numpy(),abs(out.cpu().detach().numpy() - truth),'-',label='spectrum')
    ax3.set_xlabel('$\omega$')
    ax3.set_ylabel('absolute error')
    ax3.grid(axis='both', alpha=.3)
    ax3.legend()

    ax4.plot(taui.cpu().detach().numpy(),abs(target.cpu().detach().numpy() - output.cpu().detach().numpy()),'-',label='correlator')
    ax4.set_xlabel('$\\tau$') 
    ax4.set_ylabel('absolute error')  
    ax4.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax4.grid(axis='both', alpha=.3)
    ax4.legend()

    plt.show()
    # fig.savefig('{}/figures/nnrho_figure{}_noise{}_l2{:.1E}_s{:.1E}_w{}_d{}.txt'.format(method,args.Index,args.noise,args.l2,args.slambda,args.width,args.depth))


def saveresults(method,omegai,target,out,output):

    np.savetxt('{}/Dtau/Dtau{}_noise{}_l2{:.1E}_s{:.1E}_w{}_d{}.txt'.format(method,args.Index,args.noise,args.l2,args.slambda,args.width,args.depth)\
        , np.column_stack((target.cpu().detach().numpy(),output.cpu().detach().numpy())),fmt='%8f %.10f') 
    np.savetxt('{}/rho/rho{}_noise{}_l2{:.1E}_s{:.1E}_w{}_d{}.txt'.format(method,args.Index,args.noise,args.l2,args.slambda,args.width,args.depth),\
         np.column_stack((omegai.cpu().detach().numpy(),out.cpu().detach().numpy())), fmt='%8f %.10f') 

    # PATH =
    # torch.save(nnrho, PATH)