import cv2
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class TopicRecommender:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.scaler = StandardScaler()
    
    def train(self, performance_data, topics):
        X = np.array(performance_data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, topics)
    
    def recommend(self, performance):
        X = np.array(performance).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]

class OCR:
    def __init__(self):
        digits = datasets.load_digits()
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(digits.data, digits.target)

    def process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def find_digits(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def recognise(self, contours, original_image):
        recognized_digits = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            digit = original_image[y:y + h, x:x + w]
            digit = cv2.resize(digit, (8, 8))
            digit = digit.flatten() / 16.0
            digit = np.array([digit])
            
            prediction = self.knn.predict(digit)
            recognized_digits.append((prediction[0], (x, y, w, h)))
        
        return recognized_digits

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image from path: {image_path}")
            return
        
        print("Image loaded successfully.")
        processed_image = self.process(image)
        
        contours = self.find_digits(processed_image)
        recognized_digits = self.recognise(contours, processed_image)
        
        for digit, (x, y, w, h) in recognized_digits:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Recognized Digits', image)
        print("OCR Reading Successful.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return ''.join(str(digit[0]) for digit in recognized_digits)

class QuizSystem:
    def __init__(self):
        self.topic_recommender = TopicRecommender()
        self.ocr_processor = OCR()
        self.student_history = {}

    def train_models(self, topic_data, topic_labels):
        self.topic_recommender.train(topic_data, topic_labels)
        print("Models trained successfully")

    def process_multiple_quizzes(self, student_id, quiz_files):
        results = {}
        
        for subject, file_path in quiz_files.items():
            result = self.process_quiz(student_id, file_path, subject)
            results[subject] = result
        
        scores = {subject: result['score'] for subject, result in results.items()}
        lowest_subject = min(scores, key=scores.get)
        
        for result in results.values():
            result['topic_recommendation'] = lowest_subject
            
        overall_score = np.mean([result['score'] for result in results.values()])
        
        return {
            'individual_results': results,
            'overall_score': overall_score,
            'recommended_focus': lowest_subject
        }

    def process_quiz(self, student_id, quiz_image_path, subject):
        recognized_answer = self.ocr_processor.process_image(quiz_image_path)
        score = self.calculate_score(recognized_answer)

        if student_id not in self.student_history:
            self.student_history[student_id] = {}
        if subject not in self.student_history[student_id]:
            self.student_history[student_id][subject] = []
        self.student_history[student_id][subject].append(score)

        return {
            'student_id': student_id,
            'subject': subject,
            'score': score,
            'recognized_answer': recognized_answer,
            'topic_recommendation': None  # This will be set after all quizzes are processed
        }

    def calculate_score(self, recognized_answer):
        try:
            return sum(int(digit) for digit in recognized_answer if digit.isdigit())
        except ValueError:
            print(f"Error calculating score for recognized answer: {recognized_answer}")
            return 0

if __name__ == "__main__":
    quiz_system = QuizSystem()

    quiz_system.train_models(
        [
            [28, 25, 27],
            [16, 15, 17],
            [8, 7, 9],
            [25, 28, 26],
            [12, 10, 11],
            [5, 6, 4]
        ],
        ['math', 'science', 'history', 'math', 'science', 'history']
    )

    quiz_files = {
        'math': r'C:\Users\Naysa\Documents\assignments\website_quiz\math_quiz.png',
        'science': r'C:\Users\Naysa\Documents\assignments\website_quiz\science_quiz.png',
        'history': r'C:\Users\Naysa\Documents\assignments\website_quiz\history_quiz.png'
    }

    results = quiz_system.process_multiple_quizzes("12345", quiz_files)

    print("\nOverall Results")
    print(f"Overall Score: {results['overall_score']:.2f}")
    
    print("\nIndividual Subject Results:")
    for subject, result in results['individual_results'].items():
        print(f"\n{subject.upper()} Quiz Results:")
        print(f"Score: {result['score']}")
        print(f"Recognized Answer: {result['recognized_answer']}")
        print(f"Topic Recommendation: {result['topic_recommendation']}")
