from leastqr import leastsq
import numpy as np


#Condiciones para factorizacion QR:
#   A de rango completo
#   m >= n
#   dim(b) = m

def test():
    # Test bank: A, b
    testCases = (
    #4x2
    (np.array([[5.1, 0], [3.4, 1], [-4, 1], [0.11, 1]]), np.array([[5, 1.8, 9.9, -1.5]])), 
    #3x1 
    (np.array([[-1], [0], [-1]]), np.array([[1, 5, 2]])),          
    #3x3                           
    (np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]), np.array([[0.7, 3, 4.1]])),        
    #4x2        rank(A)<n         
    (np.array([[1, 0], [1, 0], [1, 0], [1, 0]]), np.array([[5, 1.8, 9.9, -1.5]])),     
    #2x3        m<n     
    (np.array([[4, 1, 1], [1, 0, 1]]), np.array([[1, 2, -4]])),                   
     #2x3        m<n     rank(A)<n            
    (np.array([[1, 1, 1], [1, 1, 1]]), np.array([[1, 2, -4]])),     
    #4x2        Dimensions of A (mxn) and b (nx1) mismatch                         
    (np.array([[5.1, 0], [3.4, 1], [-4, 1], [0.11, 1]]), np.array([[5, 1.8, 9.9]])),  
    #4x2        b with more columns       
    (np.array([[5.1, 0], [3.4, 1], [-4, 1], [0.11, 1]]), np.array([[5, 1.8, 9.9, -1.5], [1, 0, 0, 1]])),        
    )

    eps = 10**(-6)
    passed = failed = 0

    print("___________________________________________")
    print("TEST BENCH")

    for i in range(len(testCases)):
        A = testCases[i][0]
        b = testCases[i][1]
        
        #b = testCases[n] if n < n_casos_particulares else createAb()
        print("TEST N°: ", i)
        print("A =")
        print(A)
        print("b =")
        print(b)
        
        x = leastsq(A, b.T)

        if x.size == 0:
            (m,n) = A.shape
            (o,p) = b.shape
            if m < n or np.linalg.matrix_rank(A) < n or m != o or p != 1:
                print("ISSUE DETECTED. PASSED \n")
                passed +=1
            else:
                print("ISSUE NOT DETECTED. FAILED \n")
                failed += 1
        else:
            print("x =")
            print(x)
            x_prime = np.linalg.lstsq(A,b.T, rcond=None)
            print("x_prime =")
            print(x_prime[0])
            diff = np.linalg.norm(x - x_prime[0])

            if diff < eps:
                print("norm(x-x_prime) < eps. PASSED \n")
                passed +=1
            else:
                print("norm(x-x_prime) >= eps. FAILED \n")
                failed +=1

    print("CASES: ", len(testCases))
    print("PASS: ", passed)
    print("FAIL: ", failed)

print(test())
