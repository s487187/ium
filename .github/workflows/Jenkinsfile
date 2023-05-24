pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile'
            args '-u root:root'
            registryUrl 'https://hub.docker.com/repository/docker/s487187/ium'
        }
    }

    stages {
        stage('Preparation') {
            when { expression { true } }
            steps {
                script {
                    try {
                        properties([
                            parameters([
                                // string(
                                //     defaultValue: '',
                                //     description: 'Kaggle username',
                                //     name: 'KAGGLE_USERNAME',
                                //     trim: false
                                // ),
                                // password(
                                //     defaultValue: '',
                                //     description: 'Kaggle token taken from kaggle.json file',
                                //     name: 'KAGGLE_KEY'
                                // ),
                                string(
                                    defaultValue: '50',
                                    description: 'number of examples in dataset',
                                    name: 'CUTOFF'
                                )
                            ])
                        ])
                    } catch (err) {
                        error "Failed to set up parameters: ${err.message}"
                    }
                }
            }
        }
        stage('Build') {
            steps {
                script {
                    try {
                        sh 'rm -rf ium_487187' 
                        sh '''
                            #!/bin/bash
                            pip install kaggle

                            git clone https://git.wmi.amu.edu.pl/s487187/ium_487187.git

                            echo "Processed Data" > output.txt
                        '''
                        sh "head -n ${params.CUTOFF} data.csv"
                    } catch (err) {
                        error "Failed to build: ${err.message}"
                    }
                }
            }
        }

        stage('End') {
            // when { expression { params.KAGGLE_USERNAME && params.KAGGLE_KEY } }
            steps {
                echo 'Program ended!'
            }
        }

        stage('Archive Artifact') {
            steps {
                archiveArtifacts 'output.txt'
            }
        }
    }
}
