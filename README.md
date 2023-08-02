# Real Estate Price Prediction in France

A comprehensive project by Théo Gachet, aiming to predict real estate prices in France using various property attributes.

---

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project was designed to predict the price of real estate properties based on several attributes: number of rooms, crime rate, and level of education in the area. Special emphasis has been given to express the uncertainty using 95% prediction intervals.

---

## Dependencies

- Python 3.x
- Matplotlib
- Numpy
- Pandas
- PyTorch
- Scikit-learn
- UQ360

---

## Dataset Structure

The dataset is generated with properties having attributes such as:

- Number of rooms (`nb_rooms`): A discrete value with a marginal Gaussian distribution ranging from 1 to 10.
- Crime rate (`crime_rate`): A continuous value between 0 and 1, representing the crime rate in the area.
- Education rate (`education_rate`): A continuous value between 0 and 1, representing the quality of education in the vicinity.

The target variable is the price of the property (`y`), normalized to reflect the average real estate price in France.

---

## Usage

1. **Setup Virtual Environment (Optional)**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Prediction Model**
    ```bash
    python main.py
    ```

---

## Contributing

Contributions are always welcome! Please see the `CONTRIBUTING.md` file for guidelines on contributing to this project.

---

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more information.

---

For questions, issues, or feedback, please raise an issue in the repository or contact the author, Théo Gachet.

---
