from setuptools import find_packages, setup


HYPHEN_E_DOT = 'e .'

def get_requirements(file:str)->list[str]:
    """
    The function will return list of requirements
    """
    requirements = []
    with open(file) as requirement_file:
        requirements=requirement_file.readlines()
        requirements=[requirement.replace('\n', '') for requirement in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    
    return requirements


setup(
    name='ec-crime-workshop',
    version='0.1',
    author='Saquib Mohiuddin Siddiqui',
    author_email='ssiddiqui@educationcubed.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
            