from torch.autograd import grad
import torch

def gradient(T, x, i=0, j=0, keep_gradient=False):
    """Calcul le gradient de dTi/dxj

    Args:
        T (_type_): la température calculée avec NN
        x (_type_): l'input
        i (_type_): Quelle composante (ici 0)
        j (_type_): par rapport à quelle variable
        keep_gradient : pour savoir si pytorch enregistre ce gradient 
        (si on a besoin d'une dérivée plus haute)
    """
    return grad(T[:,i], x, create_graph=keep_gradient, grad_outputs=torch.ones_like(T[:,i]))[0][:,j]

def derivee_seconde(T, x_input, j ):
    """calcul d2T/d2xj

    Args:
        T (_type_): la fonction
        x (_type_): l'input
        j (_type_): par rapport à quelle variable
    """
    T_grad = gradient(T,x_input,i=0, j=j, keep_gradient=True)
    return grad(T_grad, x_input, create_graph=True, grad_outputs=torch.ones_like(T_grad))[0][:,j]
    