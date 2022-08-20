from setuptools import find_packages, setup

setup(
      name='limbic',
      version='0.4.1',
      description='Python package for emotion analysis from text',
      long_description=open('README.rst').read(),
      url='https://github.com/glhuilli/limbic',
      author="Gaston L'Huillier",
      author_email='glhuilli@gmail.com',
      license='MIT License',
      packages=find_packages(),
      package_data={
            '': ['README.rst', 'LICENSE', 'models/*']
      },
      zip_safe=False,
      install_requires=[x.strip() for x in open("requirements.txt").readlines()])
