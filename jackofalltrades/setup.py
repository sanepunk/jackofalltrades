from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.txt").read_text()

setup(
    name = "jackofalltrades",
    packages=find_packages(exclude=["*.tests", "*.txt", "test"]),
    long_description=long_description,
    long_description_content_type="text/plain",
    description = """
        Jack of All Trades: A Simple and User-Friendly Machine Learning Toolkit

        Tired of complex machine learning libraries? Introducing jackofalltrades, a streamlined Python package designed to make machine learning accessible for everyone.

        **What sets jackofalltrades apart?**

        - Simplicity: We prioritize clear, concise functions with minimal parameters and intuitive interfaces.
        - Ease of Use: Get started quickly with our well-documented functions and a focus on straightforward implementation.
        - Core Machine Learning Algorithms: Built-in implementations for essential algorithms like linear regression, classification models (coming soon!), and more.
        - Compatibility: Seamlessly work with data formats used by popular libraries like scikit-learn.

        **Ideal for:**

        - Beginners eager to learn machine learning fundamentals.
        - Experienced users seeking a simpler alternative for quick experimentation.
        - Educators looking for a user-friendly teaching tool.

        **Get Started Now!**

        1. Install jackofalltrades using pip: `pip install jackofalltrades`
        2. Import the library and explore its functions:

        ```python
        import jackofalltrades as joft
        from jackofalltrades.datasets import get_data
        # Load data and split into training and testing sets
        ldset = get_dataset()
        X, y = ldset.get_btc()
        # Train a linear regression model
        model = joft.Models.LinearRegression(X, y)
        model.fit()
        # Make predictions and evaluate performance
        y_predicted = model.predict(X)
        model.evaluate(y, y_predicted)
        ```

        Embrace simplicity and unlock the power of machine learning with jackofalltrades!
        """,
    author = "Sane Punk",
    author_email = "punk00pp@gmail.com",
    url = "https://github.com/lazy-punk/jackofalltrades",  # Replace with your project URL
    version = "0.0.1",
    install_requires=['pandas >= 2.2.2',
                      'numpy >= 1.26.4',
                      'scikit-learn >= 1.5.0',
                      'matplotlib >= 3.9.0',
                      'jax[cpu] >= 0.4.28',
                      'tqdm >= 4.66.4'                     
                      ],
    package_data={'jackofalltrades': ['datasets/*.csv','*.txt']},
    include_package_data=True,
    test_suite='test',
    license = "MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',  # Adjust development status as needed
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Machine Learning'],)