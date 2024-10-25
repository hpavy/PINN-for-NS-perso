import torch
from pyDOE import lhs


class Rectangle:
    def __init__(self, x_max, y_max, t_min, t_max, x_min=0, y_min=0):
        """on crée ici un rectangle

        Args:
            x_max (_type_): la taille en x maximale
            y_max (_type_): la taille en y maximale
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max

    def generate_border(self, n):
        """génère n valeurs randoms sur les bords du rectangle"""
        if n % 4 != 0:
            raise ValueError("mettre n divisible par 4")  # A changer plus tard
        points = torch.zeros(n, 2)
        points[: n // 4] = torch.stack(
            (
                self.x_min * torch.ones(n // 4),
                (self.y_max - self.y_min) * torch.rand(n // 4),
            ),
            dim=1,
        )
        points[n // 4 : n // 2] = torch.stack(
            (
                self.x_max * torch.ones(n // 4),
                (self.y_max - self.y_min) * torch.rand(n // 4),
            ),
            dim=1,
        )
        points[n // 2 : 3 * n // 4] = torch.stack(
            (
                (self.x_max - self.x_min) * torch.rand(n // 4),
                self.y_max * torch.ones(n // 4),
            ),
            dim=1,
        )
        points[3 * n // 4 :] = torch.stack(
            (
                (self.x_max - self.x_min) * torch.rand(n // 4),
                self.y_min * torch.ones(n // 4),
            ),
            dim=1,
        )
        return torch.cat(
            (points, torch.rand(n, 1) * (self.t_max - self.t_min)), dim=1
        ).requires_grad_()

    def generate_random(self, n, init=False):
        """génère n valeurs randoms dans le rectangle avec un temps aléatoire,
        si init est True, alors le temps est initialisé à 0
        """
        points = torch.stack(
            (
                (self.x_max - self.x_min) * torch.rand(n) + self.x_min,
                (self.y_max - self.y_min) * torch.rand(n) + self.y_min,
            ),
            dim=1,
        )
        if not init:
            return torch.cat(
                (points, torch.rand(n, 1) * (self.t_max - self.t_min) + self.t_min),
                dim=1,
            ).requires_grad_()
        else:
            return torch.cat((points, torch.zeros(n, 1)), dim=1).requires_grad_()

    def generate_lhs(self, n):
        """Donne répartition latin hypercube"""
        max_min = torch.tensor(
            [self.x_max - self.x_min, self.y_max - self.y_min, self.t_max - self.t_min]
        )
        minn = torch.tensor([self.x_min, self.y_min, self.t_min])
        tensor_final = minn + max_min * torch.from_numpy(lhs(3, n))
        return tensor_final.to(dtype=torch.float32).requires_grad_()


if __name__ == "__main__":
    test = Rectangle(x_max=5, y_max=2, t_min=0, t_max=15)
    print(test.generate_lhs(96).dtype)
