using StatsKit

function processData(data::DataFrame)::DataFrame
    data = select(data, Not([:PassengerId, :Name]))
    data.RoomDeck = categorical([String(deck[1]) for deck in split.(data.Cabin, "/")])
    data.RoomNum = [String(deck[2]) for deck in split.(data.Cabin, "/")]
    data.RoomSide = categorical([String(deck[3]) for deck in split.(data.Cabin, "/")])
    data.TotalSpent = data.RoomService + data.FoodCourt + data.ShoppingMall + data.Spa + data.VRDeck
    data = select(data, Not(:Cabin))
    data = DataFrames.transform(data, names(data, Multiclass) .=> coerce, renamecols=false)

    return data
end

data = CSV.read("data/train.csv", DataFrame) |> 
    dropmissing |> 
    processData


###Stats Model
fn = @formula(Transported ~ HomePlanet + CryoSleep)
logitModel = glm(fn, data, Bernoulli(), LogitLink())

function measureAccuracy(model, data::DataFrame)::Float64
    predictions = GLM.predict(model, data)
    predictions = [
        if x < 0.5
            false
        else
            true
        end for x in predictions
    ]
    prediction_df = DataFrame(y_actual=data.Transported, y_predicted=predictions, prob_predicted=predictions)
    prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted
    accuracy = mean(prediction_df.correctly_classified)

    return accuracy

end

acc = measureAccuracy(logitModel, data)