import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from scipy.sparse import hstack
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, 0, 1)

        for i in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)

    def _activation(self, x):
        return np.where(x >= 0, 1, 0)

def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def print_evaluation_report(y_true, y_pred, set_name="Test"):
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"{set_name} Set Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted 0  Predicted 1")
    print(f"Actual 0        {metrics['confusion_matrix'][0, 0]:>11}  {metrics['confusion_matrix'][0, 1]:>11}")
    print(f"Actual 1        {metrics['confusion_matrix'][1, 0]:>11}  {metrics['confusion_matrix'][1, 1]:>11}")
    print(f"{'='*50}\n")
    
    return metrics

def test_perceptron_and_gate():
    print("\n" + "="*70)
    print("TESTING PERCEPTRON ON AND GATE")
    print("="*70)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)
    print("Actual:     ", y)
    print("Accuracy:   ", np.mean(predictions == y) * 100, "%")
    print(f"\nLearned weights: {perceptron.weights}")
    print(f"Learned bias: {perceptron.bias}")

def evaluate_perceptron_synthetic():
    print("\n" + "="*70)
    print("EVALUATING PERCEPTRON ON SYNTHETIC DATA")
    print("="*70)
    
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
    perceptron.fit(X_train, y_train)
    y_train_pred = perceptron.predict(X_train)
    y_test_pred = perceptron.predict(X_test)
    train_metrics = print_evaluation_report(y_train, y_train_pred, "Training")
    test_metrics = print_evaluation_report(y_test, y_test_pred, "Test")
    
    # Test with multiple random splits
    print("\nTesting with multiple random splits:")
    accuracies = []
    for seed in range(10):
        X, y = make_classification(
            n_samples=200, n_features=2, n_informative=2, 
            n_redundant=0, n_clusters_per_class=1,
            class_sep=2.0, random_state=seed
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
        perceptron.fit(X_train, y_train)
        y_pred = perceptron.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)
        print(f"Seed {seed}: Test Accuracy = {acc:.4f}")

    print(f"\nMean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Deviation: {np.std(accuracies):.4f}")

def spam_classification():
    print("\n" + "="*70)
    print("SPAM CLASSIFICATION")
    print("="*70)
    
    # Load dataset
    spambase = fetch_ucirepo(id=94)
    X = spambase.data.features
    y = spambase.data.targets
    
    # Select features
    selected_features = ['word_freq_free', 'word_freq_credit', 'word_freq_money', 
                        'capital_run_length_average', 'word_freq_receive']
    X_selected = X[selected_features]
    
    # Split data (80-10-10)
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev, y_train_dev, test_size=0.125, random_state=42
    )
    
    print(f"Training set: {len(X_train)}")
    print(f"Development set: {len(X_dev)}")
    print(f"Test set: {len(X_test)}")
    
    # Train SVM (now first)
    print("\n--- SVM ---")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train.values.ravel())
    
    y_dev_pred_svm = svm.predict(X_dev)
    svm_dev_metrics = print_evaluation_report(y_dev.values.ravel(), y_dev_pred_svm, 
                                             "SVM - Development")
    
    y_test_pred_svm = svm.predict(X_test)
    svm_test_metrics = print_evaluation_report(y_test.values.ravel(), y_test_pred_svm, 
                                              "SVM - Test")
    
    # Train Logistic Regression (now second)
    print("\n--- Logistic Regression ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train.values.ravel())
    
    y_dev_pred_logreg = log_reg.predict(X_dev)
    logreg_dev_metrics = print_evaluation_report(y_dev.values.ravel(), y_dev_pred_logreg, 
                                                 "Logistic Regression - Development")
    
    y_test_pred_logreg = log_reg.predict(X_test)
    logreg_test_metrics = print_evaluation_report(y_test.values.ravel(), y_test_pred_logreg, 
                                                  "Logistic Regression - Test")
    
    # Train Perceptron (now last)
    print("\n--- Perceptron ---")
    perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
    perceptron.fit(X_train.values, y_train.values.ravel())
    
    y_dev_pred_perceptron = perceptron.predict(X_dev.values)
    perceptron_dev_metrics = print_evaluation_report(y_dev.values.ravel(), y_dev_pred_perceptron, 
                                                     "Perceptron - Development")
    
    y_test_pred_perceptron = perceptron.predict(X_test.values)
    perceptron_test_metrics = print_evaluation_report(y_test.values.ravel(), y_test_pred_perceptron, 
                                                      "Perceptron - Test")
    
    # Comparison table (reordered)
    results = pd.DataFrame({
        'Model': ['SVM', 'Logistic Regression', 'Perceptron'],
        'Dev Accuracy': [
            svm_dev_metrics['accuracy'],
            logreg_dev_metrics['accuracy'],
            perceptron_dev_metrics['accuracy']
        ],
        'Dev Precision': [
            svm_dev_metrics['precision'],
            logreg_dev_metrics['precision'],
            perceptron_dev_metrics['precision']
        ],
        'Dev Recall': [
            svm_dev_metrics['recall'],
            logreg_dev_metrics['recall'],
            perceptron_dev_metrics['recall']
        ],
        'Dev F1': [
            svm_dev_metrics['f1'],
            logreg_dev_metrics['f1'],
            perceptron_dev_metrics['f1']
        ],
        'Test Accuracy': [
            svm_test_metrics['accuracy'],
            logreg_test_metrics['accuracy'],
            perceptron_test_metrics['accuracy']
        ],
        'Test Precision': [
            svm_test_metrics['precision'],
            logreg_test_metrics['precision'],
            perceptron_test_metrics['precision']
        ],
        'Test Recall': [
            svm_test_metrics['recall'],
            logreg_test_metrics['recall'],
            perceptron_test_metrics['recall']
        ],
        'Test F1': [
            svm_test_metrics['f1'],
            logreg_test_metrics['f1'],
            perceptron_test_metrics['f1']
        ]
    })
    
    print("\n" + "="*90)
    print("MODEL COMPARISON - SPAM CLASSIFICATION")
    print("="*90)
    print(results.to_string(index=False))
    print("="*90)

def load_language_data(english_path, dutch_path):
    with open(english_path, 'r', encoding='utf-8') as f:
        english_sentences = [line.strip() for line in f if line.strip()]
    
    with open(dutch_path, 'r', encoding='utf-8') as f:
        dutch_sentences = [line.strip() for line in f if line.strip()]
    
    sentences = english_sentences + dutch_sentences
    labels = [0] * len(english_sentences) + [1] * len(dutch_sentences)
    return sentences, np.array(labels)

def language_classification():
    print("\n" + "="*70)
    print("LANGUAGE CLASSIFICATION (ENGLISH VS DUTCH)")
    print("="*70)
    
    # Load training data
    english_path = 'universal-declaration/english.txt'
    dutch_path = 'universal-declaration/dutch.txt'
    sentences_train, y_lang_train = load_language_data(english_path, dutch_path)
    
    # Create vectorizers
    char_vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        max_features=1000,
        lowercase=True
    )
    
    word_vectorizer = CountVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=500,
        lowercase=True
    )
    
    # Transform training data
    X_char_train = char_vectorizer.fit_transform(sentences_train)
    X_word_train = word_vectorizer.fit_transform(sentences_train)
    X_combined_train = hstack([X_char_train, X_word_train])
    
    print(f"Combined feature matrix shape: {X_combined_train.shape}")
    
    dev_english_sentences = [
        "The weather forecast predicts rain for tomorrow afternoon.",
        "Students must complete their assignments before Friday.",
        "She decided to learn a new musical instrument this year.",
        "The local farmers market opens every Saturday morning.",
        "Technology has transformed the way we communicate daily.",
        "Many people prefer reading physical books over digital ones.",
        "The professor explained the theory during today's lecture.",
        "Our neighborhood organized a cleanup event last weekend.",
        "Scientists discovered a new species in the ocean depths.",
        "The company announced plans to expand its operations globally.",
        "Children need sufficient sleep for proper growth and development.",
        "The historical building was restored to its original condition.",
        "Fresh ingredients make a significant difference in cooking quality.",
        "Several countries signed an important environmental agreement yesterday.",
        "The musician performed at venues across three different continents.",
        "Modern architecture combines functionality with aesthetic appeal beautifully.",
        "Volunteers distributed supplies to families affected by the disaster.",
        "The novel explores themes of identity and belonging deeply.",
        "Regular physical activity contributes to better mental health outcomes.",
        "The exhibition features contemporary art from emerging international artists."
    ]
    
    dev_dutch_sentences = [
        "Het weerbericht voorspelt regen voor morgenmiddag.",
        "Studenten moeten hun opdrachten voor vrijdag afronden.",
        "Zij besloot dit jaar een nieuw muziekinstrument te leren.",
        "De plaatselijke boerenmarkt gaat elke zaterdagochtend open.",
        "Technologie heeft de manier waarop wij dagelijks communiceren getransformeerd.",
        "Veel mensen geven de voorkeur aan fysieke boeken boven digitale.",
        "De professor legde de theorie uit tijdens de les vandaag.",
        "Onze buurt organiseerde vorig weekend een opruimevenement.",
        "Wetenschappers ontdekten een nieuwe soort in de oceaandiepten.",
        "Het bedrijf kondigde plannen aan om wereldwijd uit te breiden.",
        "Kinderen hebben voldoende slaap nodig voor goede groei en ontwikkeling.",
        "Het historische gebouw werd hersteld naar de originele staat.",
        "Verse ingrediënten maken een significant verschil in kookkwaliteit.",
        "Verschillende landen ondertekenden gisteren een belangrijke milieuovereenkomst.",
        "De muzikant trad op in zalen verspreid over drie verschillende continenten.",
        "Moderne architectuur combineert functionaliteit prachtig met esthetische aantrekkingskracht.",
        "Vrijwilligers distribueerden voorraden aan gezinnen getroffen door de ramp.",
        "De roman verkent thema's van identiteit en verbondenheid diepgaand.",
        "Regelmatige lichaamsbeweging draagt bij aan betere mentale gezondheidsresultaten.",
        "De tentoonstelling toont hedendaagse kunst van opkomende internationale kunstenaars."
    ]
    
    test_english_sentences = [
        "The airplane landed safely despite the challenging wind conditions.",
        "Ancient civilizations developed sophisticated agricultural irrigation systems.",
        "Professional athletes train intensively for international competitions.",
        "The telescope revealed stunning details of distant galaxies.",
        "Urban planning requires careful consideration of transportation infrastructure.",
        "Marine biologists study coral reef ecosystems and their inhabitants.",
        "The committee reviewed applications from hundreds of qualified candidates.",
        "Traditional craftsmanship techniques are being preserved by dedicated artisans.",
        "Economic policies influence employment rates and consumer spending patterns.",
        "The orchestra rehearsed tirelessly for the upcoming concert performance.",
        "Renewable energy sources offer sustainable alternatives to fossil fuels.",
        "Archaeological excavations uncovered artifacts from the Bronze Age.",
        "The chef prepared a seven-course meal for the special occasion.",
        "Digital privacy concerns have increased with social media expansion.",
        "Mountain climbers face extreme conditions at high altitude elevations.",
        "The laboratory conducts research on infectious disease prevention methods.",
        "Publishers released the highly anticipated sequel earlier than expected.",
        "Wildlife conservation efforts protect endangered species from extinction.",
        "The courtroom proceedings lasted several weeks before reaching a verdict.",
        "Educational reforms aim to improve student learning outcomes significantly."
    ]
    
    test_dutch_sentences = [
        "Het vliegtuig landde veilig ondanks de uitdagende windomstandigheden.",
        "Oude beschavingen ontwikkelden geavanceerde landbouwirrigatiesystemen.",
        "Professionele sporters trainen intensief voor internationale wedstrijden.",
        "De telescoop onthulde verbluffende details van verre sterrenstelsels.",
        "Stedelijke planning vereist zorgvuldige overweging van transportinfrastructuur.",
        "Zeebiologen bestuderen koraalrifecosystemen en hun bewoners.",
        "De commissie beoordeelde aanvragen van honderden gekwalificeerde kandidaten.",
        "Traditionele ambachtelijke technieken worden bewaard door toegewijde ambachtslieden.",
        "Economisch beleid beïnvloedt werkgelegenheidspercentages en consumentenbestedingspatronen.",
        "Het orkest repeteerde onvermoeibaar voor de aankomende concertuitvoering.",
        "Hernieuwbare energiebronnen bieden duurzame alternatieven voor fossiele brandstoffen.",
        "Archeologische opgravingen onthulden artefacten uit de Bronstijd.",
        "De kok bereidde een zevengangenmenu voor de speciale gelegenheid.",
        "Zorgen over digitale privacy zijn toegenomen met de uitbreiding van sociale media.",
        "Bergbeklimmers ervaren extreme omstandigheden op grote hoogte.",
        "Het laboratorium voert onderzoek uit naar preventiemethoden voor infectieziekten.",
        "Uitgevers brachten het langverwachte vervolg eerder uit dan verwacht.",
        "Natuurbeschermingsinspanningen beschermen bedreigde diersoorten tegen uitsterven.",
        "De rechtbankprocedures duurden verschillende weken voordat een oordeel werd bereikt.",
        "Onderwijshervormingen zijn erop gericht de leerresultaten van studenten aanzienlijk te verbeteren."
    ]
    
    # Prepare development and test sets
    sentences_dev = dev_english_sentences + dev_dutch_sentences
    y_lang_dev = np.array([0] * len(dev_english_sentences) + [1] * len(dev_dutch_sentences))
    
    sentences_test = test_english_sentences + test_dutch_sentences
    y_lang_test = np.array([0] * len(test_english_sentences) + [1] * len(test_dutch_sentences))
    
    print(f"\nDevelopment set: {len(sentences_dev)} sentences")
    print(f"Test set: {len(sentences_test)} sentences")
    
    # Transform dev and test sets
    X_char_dev = char_vectorizer.transform(sentences_dev)
    X_word_dev = word_vectorizer.transform(sentences_dev)
    X_combined_dev = hstack([X_char_dev, X_word_dev])
    
    X_char_test = char_vectorizer.transform(sentences_test)
    X_word_test = word_vectorizer.transform(sentences_test)
    X_combined_test = hstack([X_char_test, X_word_test])
    
    # Train SVM (now first)
    print("\n--- SVM ---")
    svm_lang = SVC(kernel='linear', random_state=42)
    svm_lang.fit(X_combined_train, y_lang_train)
    
    y_lang_dev_pred_svm = svm_lang.predict(X_combined_dev)
    svm_lang_dev_metrics = print_evaluation_report(y_lang_dev, y_lang_dev_pred_svm, 
                                                   "SVM - Language (Development)")
    
    y_lang_test_pred_svm = svm_lang.predict(X_combined_test)
    svm_lang_test_metrics = print_evaluation_report(y_lang_test, y_lang_test_pred_svm, 
                                                    "SVM - Language (Test)")
    
    # Train Logistic Regression (now second)
    print("\n--- Logistic Regression ---")
    log_reg_lang = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_lang.fit(X_combined_train, y_lang_train)
    
    y_lang_dev_pred_logreg = log_reg_lang.predict(X_combined_dev)
    logreg_lang_dev_metrics = print_evaluation_report(y_lang_dev, y_lang_dev_pred_logreg, 
                                                      "Logistic Regression - Language (Development)")
    
    y_lang_test_pred_logreg = log_reg_lang.predict(X_combined_test)
    logreg_lang_test_metrics = print_evaluation_report(y_lang_test, y_lang_test_pred_logreg, 
                                                       "Logistic Regression - Language (Test)")
    
    # Train Perceptron (now last)
    print("\n--- Perceptron ---")
    perceptron_lang = Perceptron(learning_rate=0.01, n_iterations=100)
    perceptron_lang.fit(X_combined_train.toarray(), y_lang_train)
    
    y_lang_dev_pred_perceptron = perceptron_lang.predict(X_combined_dev.toarray())
    perceptron_lang_dev_metrics = print_evaluation_report(y_lang_dev, y_lang_dev_pred_perceptron, 
                                                          "Perceptron - Language (Development)")
    
    y_lang_test_pred_perceptron = perceptron_lang.predict(X_combined_test.toarray())
    perceptron_lang_test_metrics = print_evaluation_report(y_lang_test, y_lang_test_pred_perceptron, 
                                                           "Perceptron - Language (Test)")

if __name__ == "__main__":
    language_classification()
    spam_classification()
    evaluate_perceptron_synthetic()
    test_perceptron_and_gate()