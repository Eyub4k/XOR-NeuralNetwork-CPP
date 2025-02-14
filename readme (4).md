# Réseau de Neurones From Scratch en C++

Ce projet implémente un réseau de neurones multicouche from scratch en C++ pour résoudre le problème du XOR. L'implémentation n'utilise aucune bibliothèque externe de machine learning, seulement les bibliothèques standards C++.

## 🧠 Concept et Architecture

Le réseau est composé de :
- Couche d'entrée : 2 neurones (pour les 2 entrées du XOR)
- Couche cachée : 4 neurones
- Couche de sortie : 1 neurone

### Problème du XOR

Le XOR (OU exclusif) est un problème classique non linéairement séparable :
```
Entrée1  Entrée2  |  Sortie
   0       0      |    0
   0       1      |    1
   1       0      |    1
   1       1      |    0
```

## 🔍 Détails Techniques

### Fonction d'Activation

Initialement, la sigmoïde était utilisée comme fonction d'activation :
```cpp
double activate(double x) {
    return 1.0 / (1.0 + exp(-x)); // Sortie entre [0, 1]
}
```

Problème rencontré : La sigmoïde n'était pas adaptée car :
- Elle est centrée autour de 0.5 et non de 0
- Sa plage de sortie [0, 1] n'est pas optimale pour le XOR
- Les gradients sont plus faibles que tanh

Solution : Utilisation de tanh :
```cpp
double activate(double x) {
    return tanh(x); // Sortie entre [-1, 1]
}
```

Avantages de tanh :
- Centrée en 0
- Plage de sortie [-1, 1]
- Gradients plus forts
- Meilleure convergence pour ce type de problème

### Backpropagation

La backpropagation utilise le concept de delta (δ) :
- Pour la couche de sortie : δ = (target - output) * f'(x)
- Pour les couches cachées : δ = (Σ δ_suivant * w) * f'(x)

Les poids sont mis à jour selon la règle :
```cpp
w_new = w_old + learning_rate * delta * input
```

## 💻 Avantages du C++

1. Performance :
   - Exécution plus rapide que Python
   - Gestion directe de la mémoire
   - Pas d'overhead d'interpréteur

2. Possibilités d'Optimisation :
   - Parallélisation possible avec OpenMP ou std::thread
   - Vectorisation des calculs
   - Cache-friendly data structures

3. Contrôle Total :
   - Pas de magie noire des frameworks
   - Compréhension complète du fonctionnement
   - Personnalisation totale possible

## 🚀 Améliorations Possibles

1. Parallélisation :
   - Paralléliser le calcul des neurones dans chaque couche
   - Paralléliser le traitement des batchs pendant l'entraînement
   - Utiliser OpenMP pour une implémentation simple :
   ```cpp
   #pragma omp parallel for
   for (size_t i = 0; i < neurons.size(); i++) {
       // calculs des neurones
   }
   ```

2. Optimisations :
   - Utilisation de SIMD pour les opérations vectorielles
   - Batch training pour améliorer la convergence
   - Mini-batch stochastic gradient descent

## 📈 Résultats

Après entraînement, le réseau atteint une précision excellente :
```
Entrée: 0 0 - Sortie: 0.000309991 - Attendu: 0
Entrée: 0 1 - Sortie: 0.999058 - Attendu: 1
Entrée: 1 0 - Sortie: 0.999065 - Attendu: 1
Entrée: 1 1 - Sortie: -0.000622462 - Attendu: 0
```

## 🛠 Compilation et Utilisation

```bash
g++ -std=c++11 neural_network.cpp -o neural_network
./neural_network
```

Pour activer les optimisations :
```bash
g++ -std=c++11 -O3 neural_network.cpp -o neural_network
```

Pour la parallélisation avec OpenMP :
```bash
g++ -std=c++11 -fopenmp neural_network.cpp -o neural_network
```
