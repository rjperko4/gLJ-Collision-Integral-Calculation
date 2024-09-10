# gLJ-Collision-Integral-Calculation

This contains 2 seperate codes to calculate the collision integrals for the generalized Lennard-Jones potential. In order to perform these calculations, duplicate sample_input.csv and fill it with your own information (it can have any number of lines). Then you can run the calculations using 

python3 collision_integrals.py <input> [output]

or

python3 appx_colint.py <input> [output]

where the default output is output.csv. collision_integrals.py will take a while depending on your values, but it is the precise calculation of the collision integrals. The appx_colint.py program will import colint_11_table.csv and colint_11_table.csv to calculate a linear interpolation for a data set with 0.02 < T* < 512 and 0.5 < beta < 12. This yields a reasonable estimate for beta > 2, but if beta < 2 then the collison integral values can be somewhat erratic.

