"""
   Resource Notebook : https://www.learnpytorch.io/00_pytorch_fundamentals/    
"""
#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility import *

print(torch.__version__)

# %%

### Tensors 
### Main building block of data
#### Most of the time the Tensor itself refering to >2 dims
#### (3, 3, 3) < --- Tensor

print("################################################# TENSOR #######################################")
## Scalar 
scalar : torch.Tensor = torch.tensor(7)
print(scalar)
print("Scalar Dim %s"  % (scalar.ndim))
print("Scalar Shape %s"  % (str(scalar.shape)))
## Get the tensor back as Python INT
print(scalar.item())


#### Vector
vector : torch.Tensor =  torch.tensor([
    10, 10
])

print()
print(vector)
print("Vector Dim %s"  % (vector.ndim))
print("Vector Shape %s" %(vector.shape))


#### MATRIX
MATRIX : torch.Tensor = torch.tensor( [
    [2,7,3],
    [3,4,3],
    [3,2,3]
] )
print()
print(MATRIX)
print("Matrix dim %s" %(MATRIX.ndim))
print("Matrix shape %s" %(str(MATRIX.shape)))



#### TENSOR
# ###  Shape (2,3,2) --> Rows, columns , members 
TENSOR : torch.Tensor = torch.Tensor(
   [
    [
     [2,3],
     [2,5],
     [3,2],
     ],
     [
      [4,2],
      [2,1],  
      [5,5]
      ]
 
     ]    
)
print()
print(TENSOR)
print("Tensor dim %s" %(TENSOR.ndim))
print("Tensor shape %s" %(str(TENSOR.shape)))
print("################################################# TENSOR #######################################")
print("################################################################################################")
#%%
print("################################################# RANDOM / Zeros / ones TENSOR #######################################")
print("################################################################################################")
print(
""""
   Why does it is important :
    - You can create a starting point from random number
    - Well just for storing data, why not ?
    - Add some Randomity ?? Sure you can !
    - Anyway , the NN works
      - Random numbers -> Look at data -> update random numbers -> Look at data 
"""
)
### Image representation
TENSOR_RAND : torch.Tensor = torch.rand(size=(224,224,3))
print(TENSOR_RAND[0][0])
print("Dims %s" %(TENSOR_RAND.ndim))
print("Shape %s" %str(TENSOR_RAND.shape))

#### Ones Representation
ONES : torch.Tensor = torch.ones(size=(4,2,3))
print()
print(ONES[0])
print("Dims %s" %(ONES.ndim))
print("Shape %s" %str(ONES.shape))


#### Zeros Representation
ZEROS : torch.Tensor = torch.zeros(size=(3,3,2))
print()
print(ZEROS[0])
print("Dims %s" %(ZEROS.ndim))
print("Shape %s" %str(ZEROS.shape))

print("################################################# RANDOM TENSOR #######################################")
print("################################################################################################")

#%%
print("################################################# Tensor Datatypes #######################################")
print("################################################################################################")
print(
"""
    Datatypes is one of the 3 big errors 
    1. Tensors not right datatype
    2. Tensors not right shape
    3. Tensors not on the right device
    
    Device : where the variable stored ( CPU / GPU ) , perlu sama atau ngga return error
    requires_grad --> Basically perlu TRACK gradient atau tidak 
"""
)
randtensor : torch.Tensor = torch.rand(size=(3,4), dtype=torch.float32)
print(randtensor)
SIMPLETENSOR : torch.Tensor = torch.Tensor([[
    
    [2,3,4,3],
    [34,41,22,3],
    [3,2,11,2]
    
    ],
    [
    [21,22,11,32],
    [33,55,44,32],
    [331,221,232,2]
    ]
    
    ],
        device=None
        )

print(f"Shape : {SIMPLETENSOR.shape}")
print(f"Dtype : {SIMPLETENSOR.dtype}")
print(f"Device : {SIMPLETENSOR.device}")

print("################################################# Tensor Datatypes #######################################")
print("################################################################################################")
#%%

print("################################################# Tensor Manipulation #######################################")
print("################################################################################################")
print("""
    Basically this topic cuman manipulasi doang , most of operation is ELEMENT WISE
     - Addition
     - Substraction
     - Divison
     - Multiplication
     - Matrix multiplication
    """)

TENSOR_1D : torch.Tensor = torch.Tensor([4,4,3])
TENSOR_2D_1 : torch.Tensor = torch.Tensor( [[[3,2,1],
                                             [1,2,3]]])
TENSOR_2D_2 : torch.Tensor = torch.Tensor( [
                                            [[4,1],
                                             [2,2],
                                             [4,2]],
                                             ])

TENSOR_2D_1_s : torch.Tensor = torch.Tensor([ [3,4], [4,2]])
TENSOR_2D_1_s2 : torch.Tensor = torch.Tensor( [[5,3], [3,1]])

print(f"Addition Tensor {TENSOR_1D + 5}") # Same as [4,4,3] + [1,1,1]
print("Matrix Multiplication")
print(TENSOR_2D_1.shape) 
print(TENSOR_2D_2.shape)
print(TENSOR_2D_1 @ TENSOR_2D_2) # Dot product / Matmul

print("element Wise Matrix ")
print(TENSOR_2D_1_s + TENSOR_2D_1_s2)

print("################################################# Tensor Manipulation #######################################")
print("################################################################################################")

#%%
##
print("################################################# Tensor Aggregation #######################################")
print("################################################################################################")
print("""
        Basically just aggregate whole stuff into 1, Min max kek gitu kan,
        Mean especially
""")
vector_tensor : torch.Tensor = torch.arange(0, 100, 10)
print("Maximum value  : ", torch.max(vector_tensor))
print("Minimum value  : ", torch.min(vector_tensor))
print("Mean value  : ", torch.mean(vector_tensor.type(torch.float64)))
print("################################################# Tensor Aggregation #######################################")
print("################################################################################################")


#%%
print("################################################# Tensor Reshaping stacking Squeezing #######################################")
print("################################################################################################")

print("""       
    Pada dasarnya this concept can be shortened like so 
    Reshaping --> Merubah shape tensor menjadi desired tensor
    View --> MEmberikan pointer yang sama pada passed variabel
    Stacking $ A, B need to have same shape
        Stack --> Menumpuk tensor berdasarkan dimensi yang diberikan
        Vstack --> Menumpuk tensor secara vertikal
        Hstack --> Menumpuk tensor secara horizontal
    Squeeze --> Menghapus seluruh '1' dimension pada tensor 
    Unsqueeze --> Menambahkan '1' dimension pada tensor
    
""")

# ( 2,2,3)
TENSOR_RESHAPE : torch.Tensor = torch.tensor(
        [ [[2,3,4], [2,3,1]],
          [[3,2,1], [33,2,1]]
          ]
)
print(TENSOR_RESHAPE.shape)
## You can use reshape instead of this confusing view 
x_pointer : torch.Tensor = TENSOR_RESHAPE.view(2,3,2)
print(x_pointer.shape)
print(x_pointer)
print(TENSOR_RESHAPE)
print("RESHAPING DONE ......")

V_STACK_A : torch.tensor = torch.tensor([
                                [2],[2],[3]
                                
                                ])
V_STACK_B : torch.tensor = torch.tensor([
                                [2],[4],[2]

                                ])
print(V_STACK_A.shape)
V_STACK_AB : torch.tensor = torch.vstack((V_STACK_A, V_STACK_B))
H_STACK_AB : torch.tensor = torch.hstack((V_STACK_A, V_STACK_B))
print(V_STACK_AB)
print(H_STACK_AB)
print("V Stack shape ", V_STACK_AB.shape)
print("H Stack Shape " , H_STACK_AB.shape)

# Shape ( 1, 2, 3, 3)
SQUEEZED_TENSOR: torch.tensor = torch.tensor(
    [[
        [
            [2, 3, 4], [4, 4, 5], [3, 2, 1],
         ],
        [
            [11, 22, 12], [2, 321, 22], [22, 1, 22]
        ]
    ]]
)

print("Before : " , SQUEEZED_TENSOR.shape)
print(SQUEEZED_TENSOR)
SQUEEZED_TENSOR = torch.squeeze(SQUEEZED_TENSOR)

print("After : " , SQUEEZED_TENSOR.shape)
print(SQUEEZED_TENSOR)


print("Before Unsquuezed : " , SQUEEZED_TENSOR.shape)
print(SQUEEZED_TENSOR)
SQUEEZED_TENSOR = torch.unsqueeze(SQUEEZED_TENSOR, dim=3)

print("After Unsquuzed : " , SQUEEZED_TENSOR.shape)
print(SQUEEZED_TENSOR)

print("################################################# Tensor Reshaping stacking Squeezing #######################################")
print("################################################################################################")

#%%

print("################################################# Tensor Squeezing & Permuting  #######################################")
print("################################################################################################")
##TODO Add Squeezing & Permuting implementation  

#%%
print("################################################# INDEXING  ( Selecting data from Tensors ) #######################################")
print("################################################################################################")

x = torch.arange(1, 10).reshape(shape=(1,3,3))
print(x, x.shape)

print("Indexing the Tensor Section")

print(x[0,0,1]) # get the number 1

print(x[0,2,2]) # get the number 9

print(x[0,1,1]) #get the number  5

print(x[:, 0, :]) # get all 0 dim, and last dim, but only 0 rows

print(x[:, :, 2]) # get all 0 dim, and second dim, but only get the values for the 2 columns


print("Indexing the Tensor Section")

print("################################################# INDEXING  ( Selecting data from Tensors ) #######################################")
print("################################################################################################")

#%%

print("################################################# Numpy Arr to p.tensors ) #######################################")
print("################################################################################################")

print("""
       1. Numpy arr and pytorch tensor does not share same memory addr;:w
       2. Converting from numpy to pytorchtensor --> Default dtype64 ( From numpy ) 
       3. Likewise, pytorch -> Numpy --> Default dtype32

""")
arr : np.array = np.arange(1, 10, 1) 

print(arr, arr.dtype)


## Convert to pytorch
torchArr : torch.Tensor = torch.from_numpy(arr) ## COnverting base dtype 64
## Convert to numpy 
torchArr.type( torch.float32 ) 
npArr : np.array = torchArr.numpy() ## Cnverting to numpy dtype base the tensor

print(torchArr, torchArr.dtype)

print(npArr, npArr.dtype)


