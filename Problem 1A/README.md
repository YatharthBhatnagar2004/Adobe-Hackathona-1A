# Adobe India Hackathon - Challenge 1A: Document Structure Extraction

## Project Overview

This project implements an intelligent PDF document structure extraction system that automatically identifies and extracts titles and hierarchical headings (H1, H2, H3) from PDF documents. The solution uses a machine learning ensemble approach to accurately classify text elements based on their visual and contextual features.

## Our Journey & Approach

### Initial Challenges & Iterations

Our journey began with exploring various PDF text extraction libraries including **PyMuPDF (fitz)**, **pdfplumber**, **pdf2image**, and **Tesseract OCR**. While these libraries provided basic text extraction, they lacked the intelligence to understand document structure and hierarchy.

### Dataset Creation Strategy

Facing the challenge of creating a diverse training dataset, we developed an innovative approach:

1. **Automated PDF Generation**: Created a Python script to generate diverse PDF documents with known structure
2. **Ground Truth Creation**: Generated corresponding JSON outputs for each PDF with proper heading hierarchy
3. **Data Balancing**: Addressed the critical issue of label imbalance where "Body" text vastly outnumbered heading labels
4. **Format Diversity**: Incorporated various PDF formats, layouts, and structures to ensure robust model training

### Model Development Evolution

Our model architecture evolved through several iterations:

1. **Single Algorithm Phase**: Started with basic Logistic Regression and XGBoost
2. **Ensemble Development**: Progressed to ensemble methods combining multiple algorithms
3. **Final Architecture**: Implemented a sophisticated stacking ensemble with:
   - **CatBoost**: Gradient boosting with categorical features
   - **LightGBM**: Light gradient boosting machine
   - **XGBoost**: Extreme gradient boosting (two variants)
   - **Logistic Regression**: Final meta-learner for intelligent aggregation

### Final Model Architecture

```
Input PDF → Feature Extraction → Base Models → Meta-Learner → JSON Output
                ↓
        [CatBoost, LightGBM, XGBoost1, XGBoost2] → Logistic Regression
```

The **Logistic Regression meta-learner** acts as the final decision maker, intelligently combining predictions from all base models to produce accurate heading classifications.

## Technical Implementation

### Libraries & Dependencies

- **PDF Processing**: PyMuPDF (fitz), pdfplumber
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: pandas, numpy
- **Feature Engineering**: Custom feature extraction pipeline
- **Model Persistence**: joblib

### Feature Engineering

Our feature extraction pipeline analyzes:
- **Font characteristics**: Size, weight, family
- **Layout features**: Position, spacing, alignment
- **Text properties**: Length, case, punctuation
- **Contextual features**: Relative positioning, surrounding text

### Model Training Process

1. **Data Preprocessing**: Extract features from PDF text elements
2. **Label Encoding**: Convert heading levels to numerical labels
3. **Model Training**: Train individual base models
4. **Ensemble Creation**: Combine models using stacking
5. **Meta-Learning**: Train logistic regression on base model outputs

## Performance & Constraints

- **Model Size**: ~80MB (well within 200MB limit)
- **Execution Time**: <10 seconds for 50-page PDFs
- **Accuracy**: High precision and recall for heading detection
- **Offline Operation**: No internet dependencies

## How to Build and Run

### Prerequisites

- Docker installed
- Git repository cloned

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t adobe-hackathon-1a:latest .
```

### Running the Solution

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none adobe-hackathon-1a:latest
```

### Expected Directory Structure

```
Project/
├── input/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── output/
│   ├── document1.json
│   ├── document2.json
│   └── ...
├── models/
│   ├── ensemble_pipeline.pkl
│   └── label_encoder.pkl
├── src/
│   ├── main.py
│   ├── train.py
│   └── create_dataset.py
└── Dockerfile
```

### Output Format

The solution generates JSON files in the following format:

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Background", "page": 2},
    {"level": "H3", "text": "Historical Context", "page": 3}
  ]
}
```

## Key Innovations

1. **Intelligent Dataset Generation**: Automated creation of diverse training data
2. **Advanced Ensemble Learning**: Multi-algorithm stacking for robust predictions
3. **Feature-Rich Extraction**: Comprehensive analysis of text and layout properties
4. **Scalable Architecture**: Modular design for easy extension and maintenance

## Future Enhancements

- Multilingual support for Japanese and other languages
- Enhanced handling of complex document layouts
- Real-time processing capabilities
- Integration with document management systems

---

*This solution demonstrates the power of ensemble machine learning in solving complex document understanding challenges while maintaining high performance and accuracy standards.*

