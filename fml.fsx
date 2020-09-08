module GradientDescent =

    // Note, the rate in this example is constant. Many practical optimizers use variable
    // update (rate) - often reducing.
    let rate = 0.005

    // Gradient descent
    let step f xs =   
        // Get the partial derivatives of the function
        let df xs =  fm.diff f xs  
        printfn "xs = %A" xs
        let dzx = df xs 
        // evaluate to output values 
        xs - v rate * dzx |> fm.eval

    let train f initial steps = 
        initial |> Seq.unfold (fun pos -> Some (pos, step f pos)) |> Seq.truncate steps 
        
   
