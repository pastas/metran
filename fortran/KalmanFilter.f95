      subroutine KFseq(observation, transition_matrix, transition_covariance, &
                observation_matrix, observation_variance, observation_indices,      &
                observation_count,                                                  &
                initial_state_mean, initial_state_covariance, n_timesteps, dim, dimobs, &
                sigmas, detfs, sigmacount)

      implicit none
      
      INTEGER, intent(in) :: dim, dimobs, n_timesteps
      INTEGER, intent(in) :: observation_indices(n_timesteps,dimobs)
      INTEGER, intent(in) :: observation_count(n_timesteps)
      INTEGER, intent(out) :: sigmacount
      DOUBLE PRECISION, intent(in) :: observation_variance(dimobs)
      DOUBLE PRECISION, intent(in) :: observation_matrix(dimobs,dim)
      DOUBLE PRECISION, intent(in) :: observation(n_timesteps,dimobs)
      DOUBLE PRECISION, intent(in) :: transition_matrix(dim,dim)
      DOUBLE PRECISION, intent(in) :: transition_covariance(dim,dim)
      DOUBLE PRECISION, intent(in) :: initial_state_mean(dim)
      DOUBLE PRECISION, intent(in) :: initial_state_covariance(dim,dim)
      DOUBLE PRECISION, intent(out) :: sigmas(n_timesteps), detfs(n_timesteps)
      
      INTEGER :: r,c,i,t,obsid    
      DOUBLE PRECISION :: predicted_state_mean(dim)
      DOUBLE PRECISION :: predicted_state_covariance(dim,dim)
      DOUBLE PRECISION :: filtered_state_mean(dim)
      DOUBLE PRECISION :: filtered_state_covariance(dim,dim)
      DOUBLE PRECISION :: dotmat(dim)
      DOUBLE PRECISION :: kgain(dim)
      DOUBLE PRECISION :: innovation
      DOUBLE PRECISION :: innovation_variance
      
      sigmacount = 0

      DO r=1,dim
         filtered_state_mean(r) = initial_state_mean(r)
      ENDDO
      
      DO r=1,dim
         DO c=1,dim
            filtered_state_covariance(r,c) = initial_state_covariance(r,c)
         ENDDO
      ENDDO
      
      DO t=1,n_timesteps
          sigmas(t) = 0.
      ENDDO
      DO t=1,n_timesteps
         detfs(t) = 0.
      ENDDO
      
      DO t=1, n_timesteps
         
         DO r=1,dim
            predicted_state_mean(r) = 0.
            DO c=1,dim
               predicted_state_mean(r) = predicted_state_mean(r)  &
           + transition_matrix(r,c) * filtered_state_mean(c) 
            ENDDO
         ENDDO
         
         DO c=1,dim
            DO r=1,dim
               predicted_state_covariance(r,c) = transition_matrix(r,r) &
           * filtered_state_covariance(r,c) * transition_matrix(c,c)    &
           + transition_covariance(r,c)
            ENDDO
         ENDDO      
         
         IF (observation_count(t).gt.0) sigmacount = sigmacount + 1         
         
         DO i=1,observation_count(t)
            obsid = observation_indices(t,i) + 1
           
            innovation = observation(t,obsid)
            DO r=1,dim
               innovation = innovation                             &
               - observation_matrix(obsid,r) * predicted_state_mean(r)
            ENDDO
            
            DO r=1,dim
               dotmat(r) = 0.
               DO c=1,dim
                  dotmat(r) = dotmat(r) + predicted_state_covariance(r,c) & 
                              * observation_matrix(obsid,c)
               ENDDO
            ENDDO

            innovation_variance = observation_variance(obsid)
            DO r=1,dim
               innovation_variance = innovation_variance         &
                  + observation_matrix(obsid,r) * dotmat(r)
            ENDDO
            
            DO r=1,dim
               kgain(r) = dotmat(r) / innovation_variance
            ENDDO

            DO c=1,dim
               DO r=1,dim
                  predicted_state_covariance(r,c) =           &
                     predicted_state_covariance(r,c)          &
                     - kgain(r)*kgain(c)*innovation_variance
               ENDDO
            ENDDO
            
            DO r=1,dim
               predicted_state_mean(r) = predicted_state_mean(r) & 
                + kgain(r)*innovation
            ENDDO
                 
            sigmas(sigmacount) = sigmas(sigmacount) + innovation**2 / innovation_variance
            detfs(sigmacount) = detfs(sigmacount) + log(innovation_variance) 
       
         ENDDO
         DO r=1,dim
            filtered_state_mean(r) = predicted_state_mean(r)
         ENDDO
         
         DO c=1,dim
            DO r=1,dim
               filtered_state_covariance(r,c) = predicted_state_covariance(r,c)
            ENDDO
         ENDDO
           
      ENDDO
          
      RETURN

      end subroutine KFseq


      subroutine KFuni(y,p1,q1,rmeas,x0,p0,n_timesteps,interval, &
           varray,farray)

      implicit none
      
      INTEGER, intent(in) :: n_timesteps
      DOUBLE PRECISION, intent(in) :: interval(n_timesteps)
      DOUBLE PRECISION, intent(in) :: rmeas
      DOUBLE PRECISION, intent(in) :: y(n_timesteps)
      DOUBLE PRECISION, intent(in) :: p1
      DOUBLE PRECISION, intent(in) :: q1
      DOUBLE PRECISION, intent(in) :: x0
      DOUBLE PRECISION, intent(in) :: p0
      DOUBLE PRECISION, intent(out) :: varray(n_timesteps)
      DOUBLE PRECISION, intent(out) :: farray(n_timesteps)
      
      INTEGER :: t  
      DOUBLE PRECISION :: xs , iv
      DOUBLE PRECISION :: ps
      DOUBLE PRECISION :: k
      DOUBLE PRECISION :: p1s
      DOUBLE PRECISION :: denum

     
      xs = x0
      ps = p0

      varray(1) = xs
      farray(1) = ps

      denum = 1.-p1*p1

      DO t = 2, n_timesteps
         iv = interval(t)
         xs = xs*p1**iv
         p1s = p1**(2.*iv)
         ps = q1*(1.-p1s)/denum + p1s*ps
         varray(t) = y(t)-xs
         farray(t) = ps + rmeas
         k = ps / farray(t)
         xs = xs + k*varray(t)
         ps = ps - k*k*farray(t)
      END DO
        
      RETURN

      end subroutine KFuni
