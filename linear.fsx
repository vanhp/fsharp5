module ModelExample =

    let modelSize = 10

    let checkSize = 5

    let trainSize = 500

    let validationSize = 100

    let rnd = Random()

    let noise eps = (rnd.NextDouble() - 0.5) * eps 

    /// The true function we use to generate the training data (also a linear model plus some noise)
    let trueCoeffs = [| for i in 1 .. modelSize -> double i |]

    let trueFunction (xs: double[]) = 
        Array.sum [| for i in 0 .. modelSize - 1 -> trueCoeffs.[i] * xs.[i]  |] + noise 0.5

    let makeData size = 
        [| for i in 1 .. size -> 
            let xs = [| for i in 0 .. modelSize - 1 -> rnd.NextDouble() |]
            xs, trueFunction xs |]
         
    /// Make the data used to symbolically check the model
    let checkData = makeData checkSize

    /// Make the training data
    let trainData = makeData trainSize

    /// Make the validation data
    let validationData = makeData validationSize
 
    let prepare data = 
        let xs, y = Array.unzip data
        let xs = batchOfVecs xs
        let y = batchOfScalars y
        (xs, y)

    /// evaluate the model for input and coefficients
    let model (xs: DT<double>, coeffs: DT<double>) = 
        fm.Sum (xs * coeffs, axis= [| 1 |])
           
    let meanSquareError (z: DT<double>) tgt = 
        let dz = z - tgt 
        fm.Sum (dz * dz) / v (double modelSize) / v (double z.Shape.[0].Value) 

    /// The loss function for the model w.r.t. a true output
    let loss (xs, y) coeffs = 
        let y2 = model (xs, batchExtend coeffs)
        meanSquareError y y2
          
    let validation coeffs = 
        let z = loss (prepare validationData) (vec coeffs)
        z |> fm.eval

    let train inputs steps =
        let initialCoeffs = vec [ for i in 0 .. modelSize - 1 -> rnd.NextDouble()  * double modelSize ]
        let inputs = prepare inputs
        GradientDescent.train (loss inputs) initialCoeffs steps
           
    [<LiveCheck>]
    let check1 = train checkData 1  |> Seq.last

    let learnedCoeffs = train trainData 200 |> Seq.last |> fm.toArray
         // [|1.017181246; 2.039034327; 2.968580146; 3.99544071; 4.935430581;
         //   5.988228378; 7.030374908; 8.013975714; 9.020138699; 9.98575733|]

    validation trueCoeffs

    validation learnedCoeffs
    
    