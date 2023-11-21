using LinearAlgebra, Plots



α = 1   #alpha_g
β = 0.8 #alpha_d
γ = 0.1 #alpha_b

vₐ = 1
vᵦ = 0.6

#Ud = exp(-sqrt(x*x+y*y))
function ∇Ud(x, y)
    r = -exp(-sqrt(x*x+y*y))/sqrt(x*x + y*y)
    return [x*r, y*r]
end

#Ug = sqrt(x*x + v*v)
function ∇Ug(x, y)
    v = y - 2
    r = 1/sqrt(x*x + v*v)
    return [x*r, v*r]
end

#Ub = -log(x-xl)-log(xr-x)-log(y-yl)-log(yh-y)
function ∇Ub(x,y)
    yh = -2.1; yl = -yh
    xr = 0.6; xl = -xr
    return [-1/(x-xl)+1/(xr-x),-1/(y-yl)+1/(yh-y)] 
end


function integrate(x₀, T, N)
    dt = T / (N-1); #time step
    a = zeros(2, N)
    d = zeros(2, N, 3)
    d[:,1,:] = [0 -0.5 0.5; 0  0   0]
    temp = zeros(2)
    a[:,1] = x₀
    for i ∈ 1:N-1
        temp .= 0
        for j ∈ 1:3
            temp += ∇Ud(a[1, i]-d[1,i,j], a[2, i]-d[2,i,j]) 
            r = a[:,i] - d[:,i,j]
            r += randn(2)/5
            d[:,i+1,j] = d[:,i,j] + dt * r /norm(r) * vᵦ
        end
        ∇U = α * ∇Ug(a[1, i], a[2, i]) + β * temp + γ * ∇Ub(a[1, i], a[2, i])
        v = ∇U + randn(2)/10
        a[:,i+1] = a[:,i] - dt *v/norm(v) *vₐ
    end
    return a, d
end
P = plot(framestyle=:box,xlims=(-1, 1))
for i∈1:1
    xₒ = [0.05, -2]
    a, d = integrate(xₒ, 6, 1000)
    plot!(a[1,:],a[2,:], label = false)
    
    plot!(d[1,:,1],d[2,:,1], label =false, linestyle=:dash)
    plot!(d[1,:,2],d[2,:,2], label =false, linestyle=:dash)
    plot!(d[1,:,3],d[2,:,3], label =false, linestyle=:dash)
    
end
display(P)
