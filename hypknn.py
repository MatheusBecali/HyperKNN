import numpy as np
import torch
from collections import Counter 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
# Hyperbolic Library
from hyptorch.nn import ToPoincare
from hyptorch.pmath import dist as hypdist

# ---------------------------------------------------------------------------
# Implementação do KNN Hiperbólico (Modelo de Disco de Poincaré)
# ---------------------------------------------------------------------------

class HyperbolicKNN:
    def __init__(self, k: int = 3, c: float = 0.5, validate_input_geometry: bool = True):
        """
        Inicializa o classificador KNN Hiperbólico.

        Parâmetros:
        k (int): O número de vizinhos a considerar (padrão é 3).
        c (float): Parâmetro de curvatura da bola de Poincaré (geralmente $c > 0$, onde
                   a curvatura do espaço $K = -c$ ou $K = -c^2$ dependendo da convenção.
                   Se o raio da bola é $1/\\sqrt{c_K}$ para curvatura $K > 0$, então
                   esta `c` corresponde a $c_K$. Se o raio é $1/c_R$, então esta `c` é $c_R^2$.
                   A validação usa $1/\\sqrt{c}$, então `c` é assumida como $K > 0$.
        validate_input_geometry (bool): Se True, valida se os pontos estão na bola de Poincaré.
        """
        if k <= 0:
            raise ValueError("O número de vizinhos (k) deve ser positivo.")
        if c <= 0:
            raise ValueError("O parâmetro de curvatura 'c' deve ser positivo para o modelo de Poincaré conforme usado na validação.")
        self.k = k
        self.c = c
        self.validate_input_geometry = validate_input_geometry
        self.X_train = None
        self.y_train = None

    def _validate_poincare_points(self, X: np.ndarray) -> None:
        """
        Valida se os pontos de entrada estão dentro da bola de Poincaré.

        Args:
            X (np.ndarray): Array de pontos, onde cada linha é um ponto.
            curvature (float): A curvatura do espaço hiperbólico. Deve ser positiva.

        Raises:
            ValueError: Se a curvatura não for positiva.
            AssertionError: Se algum ponto estiver fora da bola de Poincaré
                            definida por 1 / sqrt(curvature).
        """
        if self.c <= 0:
            raise ValueError("A curvatura (K) deve ser um valor positivo.")
        
        max_norm_allowed = 1 / (self.c**0.5)
        norms = np.linalg.norm(X, axis=1)

        if not np.all(norms <= max_norm_allowed):
            problematic_indices = np.where(norms > max_norm_allowed)[0]
            error_msg = (
                f"Validação da geometria falhou! Pontos com índices {problematic_indices} "
                f"estão fora da bola de Poincaré (norma > {max_norm_allowed:.4f} para K={self.c}).\n"
                f"Normas encontradas para esses pontos: {norms[problematic_indices]}"
            )
            raise AssertionError(error_msg)
        # print(f"Validação da geometria de Poincaré para {X.shape[0]} pontos bem-sucedida (K={curvature}).")
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Treina o classificador KNN Hiperbólico com os dados de treinamento.

        Parâmetros:
        X (numpy.ndarray): As características dos dados de treinamento (pontos na bola de Poincaré).
        y (numpy.ndarray): Os rótulos dos dados de treinamento.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("O número de amostras em X e y deve ser o mesmo.")
        if X.shape[0] < self.k:
             print(f"Aviso: O número de amostras de treinamento ({X.shape[0]}) é menor que k ({self.k}).")

        if self.validate_input_geometry:
            self._validate_poincare_points(X)

        self.X_train = X
        self.y_train = y

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prevê os rótulos para um conjunto de pontos de dados X_test.

        Parâmetros:
        X_test (numpy.ndarray): Os pontos de dados (na bola de Poincaré) para os quais fazer previsões.

        Retorna:
        numpy.ndarray: Um array de rótulos previstos.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("O classificador KNN deve ser treinado com o método 'fit' antes de prever.")

        if self.validate_input_geometry:
            self._validate_poincare_points(X_test)
        
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)



    def _predict_single(self, x: np.ndarray) -> int: # Renomeado para clareza e consistência
        """
        Prevê o rótulo para um único ponto de dados x na bola de Poincaré.

        Parâmetros:
        x (numpy.ndarray): O ponto de dados para o qual fazer uma previsão.

        Retorna:
        int: O rótulo previsto para o ponto de dados de entrada x.
        """
        # Calcula as distâncias hiperbólicas para todos os pontos de dados de treinamento
        distances = [hypdist(x, x_train_point, c=self.c) for x_train_point in self.X_train]

        # Obtém os índices dos k pontos de dados de treinamento mais próximos
        k_indices = np.argsort(distances)[:self.k]

        # Obtém os rótulos dos k pontos de dados de treinamento mais próximos
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Realiza a votação majoritária para determinar o rótulo final
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def main():
    # --- Configurações e Constantes ---
    SEED = 78645
    VALIDATE_INPUT_GEOMETRY = True
    CURVATURE = 0.1  # Curvatura para o espaço de Poincaré
    N_NEIGHBORS = 3  # Número de vizinhos para o KNN
    TEST_SIZE_INITIAL_SPLIT = 0.1  # Proporção para o conjunto de teste
    TEST_SIZE_VALIDATION_SPLIT = 0.1 # Proporção para o conjunto de validação (do restante)


    # Define a semente para reprodutibilidade
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- 1. Carregamento e Divisão dos Dados ---
    print("Carregando dataset Iris...")
    iris = datasets.load_iris()
    data_np, data_labels_np = iris.data, iris.target

    # Divide o dataset em treino+validação e teste
    x_train_val_np, X_test_np, y_train_val_np, y_test_np = train_test_split(
        data_np,
        data_labels_np,
        stratify=data_labels_np,
        test_size=TEST_SIZE_INITIAL_SPLIT,
        random_state=SEED,
    )

    # Divide o conjunto treino+validação em treino e validação
    X_train_np, X_valid_np, y_train_np, y_valid_np = train_test_split(
        x_train_val_np,
        y_train_val_np,
        stratify=y_train_val_np,
        test_size=TEST_SIZE_VALIDATION_SPLIT,
        random_state=SEED,
    )

    # --- 2. Conversão para Tensores PyTorch ---
    X_train = torch.Tensor(X_train_np)
    y_train = torch.Tensor(y_train_np).long()  # Rótulos como LongTensor
    X_valid = torch.Tensor(X_valid_np)
    y_valid = torch.Tensor(y_valid_np).long()  # Rótulos como LongTensor
    X_test = torch.Tensor(X_test_np)
    y_test = torch.Tensor(y_test_np).long()    # Rótulos como LongTensor

    # --- 3. Transformação para o Espaço de Poincaré e Treinamento do Modelo ---
    print(f"\nIniciando modelagem em Espaço Hiperbólico (Curvatura K={CURVATURE})...")

    # Inicializa o transformador Euclidiano para Poincaré
    e2p_transformer = ToPoincare(c=CURVATURE, train_c=False, train_x=False)

    # Mapeia os dados para a bola de Poincaré
    X_train_hyp = e2p_transformer(X_train)
    X_valid_hyp = e2p_transformer(X_valid)
    X_test_hyp = e2p_transformer(X_test)

    # Inicializa e treina o classificador KNN Hiperbólico
    hyperbolic_knn_classifier = HyperbolicKNN(k=N_NEIGHBORS, c=CURVATURE, validate_input_geometry=VALIDATE_INPUT_GEOMETRY)
    hyperbolic_knn_classifier.fit(X_train_hyp, y_train)

    # Faz predições no conjunto de teste
    y_pred_eval = hyperbolic_knn_classifier.predict(X_test_hyp)

    # --- 4. Avaliação do Modelo ---

    # Converte tensores para arrays NumPy para uso com sklearn.metrics
    y_test_eval = y_test.cpu().numpy()

    accuracy = accuracy_score(y_test_eval, y_pred_eval)
    balanced_accuracy = balanced_accuracy_score(y_test_eval, y_pred_eval)
    # average='weighted' para lidar com desbalanceamento de classes nas métricas de recall, precision, f1
    recall = recall_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
    precision = precision_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
    f1 = f1_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)

    print("\n--- Resultados da Avaliação ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Acurácia Balanceada: {balanced_accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
