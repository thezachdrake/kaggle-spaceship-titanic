using StatsKit
using MLJ
import MLJLinearModels
using AlgebraOfGraphics, CairoMakie
set_aog_theme!()


function processpassengerData(passengerData::DataFrame)::DataFrame
    passengerData = select(passengerData, Not([:PassengerId, :Name]))
    passengerData.RoomDeck = categorical([String(deck[1]) for deck in split.(passengerData.Cabin, "/")])
    #passengerData.RoomNum = [String(deck[2]) for deck in split.(passengerData.Cabin, "/")]
    passengerData.RoomSide = categorical([String(deck[3]) for deck in split.(passengerData.Cabin, "/")])
    #passengerData.TotalSpent = passengerData.RoomService + passengerData.FoodCourt + passengerData.ShoppingMall + passengerData.Spa + passengerData.VRDeck
    passengerData = select(passengerData, Not(:Cabin))
    return passengerData
end

passengerData = CSV.read("data/train.csv", DataFrame) |>
                dropmissing |>
                processpassengerData


###Data Analysis
axis = (width=225, height=225)
transported_deck = AlgebraOfGraphics.data(passengerData) * frequency() * mapping(:Transported, color=:VIP)



###Statistical Approach
fn = @formula(Transported ~ VIP + CryoSleep + Age + RoomService)
logitModel = glm(fn, passengerData, Bernoulli(), LogitLink())

function measureAccuracy(model, passengerData::DataFrame)::Float64
    predictions = GLM.predict(model, passengerData)
    predictions = [
        if x < 0.5
            false
        else
            true
        end for x in predictions
    ]
    prediction_df = DataFrame(y_actual=passengerData.Transported, y_predicted=predictions, prob_predicted=predictions)
    prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted
    accuracy = mean(prediction_df.correctly_classified)

    return accuracy

end

acc = measureAccuracy(logitModel, passengerData)


###Machine Learning Approach
scipassengerData = MLJ.coerce!(passengerData,
    MLJ.autotype(passengerData)
)

y, X = MLJ.unpack(scipassengerData, ==(:Transported); rng=123)
logi = MLJLinearModels.LogisticClassifier()
logi = MLJ.machine(logi, X, y)
train, test = MLJ.partition(eachindex(y), 0.7)

fit!(logi, rows=train)