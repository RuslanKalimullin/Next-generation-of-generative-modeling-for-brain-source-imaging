![]("https://github.com/RuslanKalimullin/Next-generation-of-generative-modeling-for-brain-source-imaging/edit/main/Logo.jpg")    
# Next generation of generative modeling for brain source imaging

[Presentation link](https://docs.google.com/presentation/d/1CO6aT8fXYmgwDTWxCvjzQ5JX0CCZnw6ixIVvrzNQDl8/edit?usp=sharing)

[Jupyter Notebook](https://github.com/RuslanKalimullin/Next-generation-of-generative-modeling-for-brain-source-imaging/blob/main/Code/CFM_torchcfm_realisation.ipynb)


# EEG-to-ECoG Inverse Problem Using Optimal Transport and Conditional Flow Matching

## 1. **Mathematical Problem Formulation**

### 1.1. **Problem Description**

The goal is to solve the **inverse problem** of recovering ECoG activity from EEG measurements. Given paired EEG and ECoG data:

- $$\( \mathbf{y}^{EEG} \in \mathbb{R}^m \)$$ — EEG measurements from $$\( m \)$$ sensors,
- $$\( \mathbf{y}^{ECoG} \in \mathbb{R}^n \)$$ — ECoG activity from $$\( n \)$$ sensors.

The task is to approximate the transformation operator $$\( \mathbf{A} \)$$ that relates the EEG measurements to ECoG activity:

$$\[
\mathbf{y}^{ECoG} = \mathbf{A} \mathbf{y}^{EEG} + \mathbf{n},
\]$$

where $$\( \mathbf{n} \)$$ represents noise. The objective is to find an approximation of $$\( \mathbf{A} \)$$ that accurately predicts ECoG activity from EEG data.

### 1.2. **Regularization via Quasi-Reversibility**

Since the problem is ill-posed, regularization is necessary to stabilize the solution. One such method is quasi-reversibility, as proposed by L. Bourgeois, which introduces a smoothness regularization term:

$$\[
\min_{\mathbf{A}} \left\| \mathbf{y}^{ECoG} - \mathbf{A} \mathbf{y}^{EEG} \right\|^2 + \alpha \left\| \nabla^2 \mathbf{y}^{ECoG} \right\|^2,
\]$$

where $$\( \alpha \)$$ is the regularization parameter, and $$\( \nabla^2 \mathbf{y}^{ECoG} \)$$ ensures the smoothness of the activity distribution.

### 1.3. **Optimal Transport for EEG-to-ECoG Mapping**

The transformation of EEG data into ECoG can be modeled using **Optimal Transport** by minimizing the Wasserstein distance between the EEG and ECoG activity distributions:

$$\[
W(\mathbf{y}^{EEG}, \mathbf{y}^{ECoG}) = \inf_{\gamma \in \Gamma(\mathbf{y}^{EEG}, \mathbf{y}^{ECoG})} \int \|\mathbf{y}^{EEG} - \mathbf{y}^{ECoG}\|^2 d\gamma(\mathbf{y}^{EEG}, \mathbf{y}^{ECoG}),
\]$$

where $$\( \gamma \)$$ is the transportation plan. This approach finds the minimal "transport cost" between EEG and ECoG activity, allowing us to approximate the transformation operator $$\( \mathbf{A} \)$$.

## 2. **Conditional Flow Matching (CFM)**

### 2.1. **CFM Description**

**Conditional Flow Matching (CFM)** is a technique designed to model continuous flows between distributions. It is particularly well-suited for modeling the complex transformations between EEG and ECoG topographies.

#### **Exact Optimal Transport Conditional Flow Matcher (OT-CFM)**

To accurately model the transition between EEG and ECoG data, we use **Exact Optimal Transport Conditional Flow Matcher**. This method is based on the exact optimal transport plan:

$$\[
z = (x_0, x_1), \quad q(z) = \pi(x_0, x_1),
\]$$

where $$\( \pi \)$$ is the optimal transport plan, and $$\( x_0 \)$$ and $$\( x_1 \)$$ represent EEG and ECoG data, respectively. This approach minimizes the Wasserstein distance between the distributions.

#### **Schrödinger Bridge Conditional Flow Matcher (SB-CFM)**

Another method is **Schrödinger Bridge Conditional Flow Matcher (SB-CFM)**, which uses an entropically regularized transport plan $$\( \pi_\epsilon \)$$. This method stabilizes training by introducing regularization, enabling the modeling of stochastic flows between EEG and ECoG data:

$$\[
z = (x_0, x_1), \quad q(z) = \pi_\epsilon(x_0, x_1),
\]$$

where $$\( \pi_\epsilon \)$$ is the regularized plan, approximated via mini-batches. This technique captures the stochastic nature of the EEG-to-ECoG transformation more accurately.

## 3. **Data-Informed Loss Functions**

To train the EEG-to-ECoG transformation model, a hybrid loss function was developed, combining several components to ensure that the model is data-driven and aligned with the biological properties of EEG and ECoG signals.

- **Laplacian Loss**: This loss minimizes the difference in the Laplacians of the predicted and true activity maps, ensuring smooth spatial distributions of activity on ECoG sensors. This aligns with the smooth nature of brain activity.
  
- **Wasserstein Loss**: By using the Wasserstein distance between EEG and ECoG activity maps, the global structure of signals and their spatial dependencies are better preserved during the transformation.

- **MSE**: The classic Mean Squared Error (MSE) ensures local accuracy between predicted and true data, controlling the pointwise error at individual sensors.

The hybrid loss function is defined as:

$$\[
\text{Loss} = \alpha \cdot \text{Wasserstein Loss} + \beta \cdot \text{MSE} + \gamma \cdot \text{Laplacian Loss},
\]$$

where $$\( \alpha \)$$, $$\( \beta \)$$, and $$\( \gamma \$$) are weighting parameters that control the contribution of each component.

---

This README describes the mathematical formulation of the EEG-to-ECoG inverse problem, the use of Conditional Flow Matching for modeling the transformation, and the data-informed loss functions used to guide the training of the model. This repository includes implementations of these methods using state-of-the-art deep learning techniques and optimal transport.

--- 
