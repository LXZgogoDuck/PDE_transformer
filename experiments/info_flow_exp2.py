import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.transformer import ImprovedTransformer
from models.pde_model import ImprovedPDEModel
from utils.visualization import visualize_states
from utils.analysis import compute_feature_correlation, plot_correlation

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

input_dim = 28 * 28
d_model = 64
nhead = 4
num_layers = 5

transformer = ImprovedTransformer(input_dim, d_model, nhead, num_layers)
pde = ImprovedPDEModel(d_model, num_steps=5, num_heads=nhead)

for exp_num in range(5):
    images, labels = next(iter(loader))
    digit = labels.item()
    inputs = images.view(1, -1)

    with torch.no_grad():
        _, transformer_states = transformer(inputs)
        pde_states = pde(transformer_states[0])

    t_corr = compute_feature_correlation(transformer_states)
    p_corr = compute_feature_correlation(pde_states)

    visualize_states(transformer_states, "Transformer Info Flow",
                     save_path=f"results/info_flow2/transformer_exp{exp_num+1}_digit{digit}.png")

    visualize_states(pde_states, "PDE Info Flow",
                     save_path=f"results/info_flow2/pde_exp{exp_num+1}_digit{digit}.png")

    plot_correlation(t_corr, p_corr, digit, exp_num+1,
                     save_path=f"results/info_flow2/correlation_exp{exp_num+1}_digit{digit}.png")

    print(f"[Experiment {exp_num+1}] Digit: {digit}")
    print("Transformer Correlation:", t_corr)
    print("PDE Correlation:", p_corr)

''' Results
[Experiment 1] Digit: 9
Transformer Correlation: [0.8788547662117059, 0.8999675428557772, 0.9444871958069854, 0.96939387094997, 0.9713215847628515]
PDE Correlation: [-0.3916221765971814, -0.6316362320260203, -0.05016858017619734, 0.7950828451825654, 0.9365813100561597]
[Experiment 2] Digit: 4
Transformer Correlation: [0.8754142112453561, 0.9346709267783089, 0.9403755233350901, 0.9416191452968126, 0.9625476555541755]
PDE Correlation: [-0.3921337266443499, -0.6436716582512952, -0.11073059016588098, 0.7329018163340774, 0.9125246136287717]
[Experiment 3] Digit: 5
Transformer Correlation: [0.8541582994413272, 0.9049060366962045, 0.9299166289082891, 0.948939042872112, 0.9489349026314653]
PDE Correlation: [-0.5253436329221595, -0.6486979092319592, -0.05094702891946298, 0.765182440085135, 0.9366700825530663]
[Experiment 4] Digit: 4
Transformer Correlation: [0.8532581367335297, 0.9192735504114194, 0.9600246139459265, 0.9620100335040125, 0.9674566327019102]
PDE Correlation: [-0.20559387366410753, -0.42146123797488805, 0.22592148665334708, 0.8632597894507371, 0.9106426448563522]
[Experiment 5] Digit: 5
Transformer Correlation: [0.8295278625309841, 0.8955242018600383, 0.9390761205803381, 0.9503801924462344, 0.9707466175210474]
PDE Correlation: [-0.474609202431722, -0.6260048549314327, -0.07846744751511368, 0.7350616153187814, 0.9082777695625857]

'''