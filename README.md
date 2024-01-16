# Linear-Algebra-for-Machine-Learning-and-Data-Science

This course is part of the [Mathematics for Machine Learning and Data Science Specialization](https://www.deeplearning.ai/courses/mathematics-for-machine-learning-and-data-science-specialization/) by [DeepLearning.AI](https://www.deeplearning.ai).

After completing this course, learners will be able to:
- Represent data as vectors and matrices and identify their properties using concepts of singularity, rank, and linear independence, etc.
- Apply common vector and matrix algebra operations like dot product, inverse, and determinants 
- Express certain types of matrix operations as linear transformations 
- Apply concepts of eigenvalues and eigenvectors to machine learning problems

### Week 1: Systems of Linear Equations
Matrices are commonly used in machine learning and data science to represent data and its transformations. In this week, you will learn how matrices naturally arise from systems of equations and how certain matrix properties can be thought in terms of operations on system of equations.

#### Learning Objectives
- Form and graphically interpret 2x2 and 3x3 systems of linear equations
- Determine the number of solutions to a 2x2 and 3x3 system of linear equations
- Distinguish between singular and non-singular systems of equations
- Determine the singularity of 2x2 and 3x3 system of equations by calculating the determinant

#### Lesson 1: Systems of Linear equations: two variables
- Machine learning motivation
- Systems of sentences
- Systems of equations
- Systems of equations as lines
- A geometric notion of singularity
- Singular vs nonsingular matrices
- Linear dependence and independence
- The determinant
- Practice Quiz: Solving systems of linear equations
- Lab: Introduction to NumPy Arrays
  
#### Lesson 2: Systems of Linear Equations: three variables
- Systems of equations (3×3)
- Singular vs non-singular (3×3)
- Systems of equations as planes (3×3)
- Linear dependence and independence (3×3)
- The determinant (3×3)
- Quiz: Matrices
- Lab: Solving Linear Systems: 2 variables

### Week 2: Solving systems of Linear Equations
In this week, you will learn how to solve a system of linear equations using the elimination method and the row echelon form. You will also learn about an important property of a matrix: the rank. The concept of the rank of a matrix is useful in computer vision for compressing images.

#### Learning Objectives
- Solve a system of linear equations using the elimination method.
- Use a matrix to represent a system of linear equations and solve it using matrix row reduction.
- Solve a system of linear equations by calculating the matrix in the row echelon form.
- Calculate the rank of a system of linear equations and use the rank to determine the number of solutions of the system.

#### Lesson 1: Solving systems of Linear Equations: Elimination
- Machine learning motivation
- Solving non-singular systems of linear equations
- Solving singular systems of linear equations
- Solving systems of equations with more variables
- Matrix row-reduction
- Row operations that preserve singularity
- Practice Quiz: Method of Elimination
- Lab: Solving Linear Systems: 3 variables

#### Lesson 2: Solving systems of Linear Equations: Row Echelon Form and Rank
- The rank of a matrix
- The rank of a matrix in general
- Row echelon form
- Row echelon form in general
- Reduced row echelon form
- Quiz: The Rank of a matrix
- Programming Assignment: System of Linear Equations

### Week 3: Vectors and Linear Transformations
An individual instance (observation) of data is typically represented as a vector in machine learning. In this week, you will learn about properties and operations of vectors. You will also learn about linear transformations, matrix inverse, and one of the most important operations on matrices: the matrix multiplication. You will see how matrix multiplication naturally arises from composition of linear transformations. Finally, you will learn how to apply some of the properties of matrices and vectors that you have learned so far to neural networks.

#### Learning Objectives
- Perform common operations on vectors like sum, difference, and dot product.
- Multiply matrices and vectors.
- Represent a system of linear equations as a linear transformation on a vector.
- Calculate the inverse of a matrix, if it exists.

#### Lesson 1: Vectors
- Norm of a vector
- Sum and difference of vectors
- Distance between vectors
- Multiplying a vector by a scalar
- The dot product
- Geometric Dot Product
- Multiplying a matrix by a vector
- Practice Quiz: Vector operations: Sum, difference, multiplication, dot product
- Lab: Vector Operations: Scalar Multiplication, Sum and Dot Product of Vectors

#### Lesson 2: Linear transformations
- Matrices as linear transformations
- Linear transformations as matrices
- Matrix multiplication
- The identity matrix
- Matrix inverse
- Which matrices have an inverse?
- Neural networks and matrices
- Quiz: Vector and Matrix Operations, Types of Matrices
- Lab: Matrix Multiplication
- Lab: Linear Transformations
- Programming Assignment: Single Perceptron Neural Networks for Linear Regression

### Week 4: Determinants and Eigenvectors
In this final week, you will take a deeper look at determinants. You will learn how determinants can be geometrically interpreted as an area and how to calculate determinant of product and inverse of matrices. We conclude this course with eigenvalues and eigenvectors. Eigenvectors are used in dimensionality reduction in machine learning. You will see how eigenvectors naturally follow from the concept of eigenbases.

#### Learning Objectives
- Interpret the determinant of a matrix as an area and calculate determinant of an inverse of a matrix and a product of matrices.
- Determine the bases and span of vectors.
- Find eigenbases for a special type of linear transformations commonly used in machine learning.
- Calculate the eignenvalues and eigenvectors of a linear transformation (matrix).

#### Lesson 1: Determinants In-depth
- Machine Learning Motivation
- Singularity and rank of linear transformation
- Determinant as an area
- Determinant of a product
- Determinants of inverses
- Practice Quiz: Determinants and Linear Transformations

#### Lesson 2: Eigenvalues and Eigenvectors
- Bases in Linear Algebra
- Span in Linear Algebra
- Interactive visualization: Linear Span
- Eigenbases
- Eigenvalues and eigenvectors
- Quiz: Eigenvalues and Eigenvectors
- Programming Assignment: Eigenvalues and Eigenvectors
